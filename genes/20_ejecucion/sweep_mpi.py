import argparse
import math
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ENERGY_RE = re.compile(r"ElectroMagnetic Field Energy:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
RAM_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time.*:\s*(.+)")


def run_and_log(cmd, cwd: Path, log_path: Path, env: Optional[dict] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            shell=False,
            env=env,
        )
    if p.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
        raise RuntimeError(
            f"Fallo comando: {' '.join(map(str, cmd))}\n"
            f"Últimas líneas de {log_path}:\n" + "\n".join(tail)
        )


def run_and_log_no_raise(cmd, cwd: Path, log_path: Path, env: Optional[dict] = None) -> Tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            shell=False,
            env=env,
        )
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return p.returncode, txt


def set_geo_vars(geo_text: str, dx_m: float, dy_m: float, dz_m: float) -> str:
    def repl(var: str, val: float, text: str) -> str:
        val_str = f"{val:.17g}"
        pattern = re.compile(rf"(?m)^\s*{re.escape(var)}\s*=\s*[-+0-9.eE]+\s*;")
        if not pattern.search(text):
            raise RuntimeError(f"No encontré '{var} = ...;' en el .geo")
        return pattern.sub(f"{var} = {val_str};", text)

    geo_text = repl("dx", dx_m, geo_text)
    geo_text = repl("dy", dy_m, geo_text)
    geo_text = repl("dz", dz_m, geo_text)
    return geo_text


def parse_energy(log_text: str) -> float:
    m = ENERGY_RE.search(log_text)
    if not m:
        raise RuntimeError("No encontré 'ElectroMagnetic Field Energy' en el log de ElmerSolver.")
    return float(m.group(1))


def parse_ram_gb(log_text: str) -> Optional[float]:
    m = RAM_RE.search(log_text)
    if not m:
        return None
    kb = int(m.group(1))
    return kb / (1024.0 * 1024.0)


def parse_elapsed_raw(log_text: str) -> Optional[str]:
    m = ELAPSED_RE.search(log_text)
    if not m:
        return None
    return m.group(1).strip()


def parse_mesh_counts_from_msh(msh_path: Path) -> Tuple[Optional[int], Optional[int]]:
    txt = msh_path.read_text(encoding="utf-8", errors="ignore")
    n_nodes = None
    n_elems = None

    m = re.search(r"\$Nodes\s+(\d+)", txt)
    if m:
        n_nodes = int(m.group(1))

    m = re.search(r"\$Elements\s+(\d+)", txt)
    if m:
        n_elems = int(m.group(1))

    return n_nodes, n_elems


def inspect_solver_log(log_text: str) -> dict:
    gcr_warn = "IterMethod_GCR: Iterated GCR solution may not be accurate" in log_text

    true_res = None
    m = re.search(r"True residual norm::\s*([0-9Ee+\-.]+)", log_text)
    if m:
        true_res = float(m.group(1))

    iter_res = None
    m = re.search(r"Iterated residual norm after\s+\d+\s+iters:\s*([0-9Ee+\-.]+)", log_text)
    if m:
        iter_res = float(m.group(1))

    gcr_iters = [int(x) for x in re.findall(r"gcr:\s+(\d+)", log_text)]
    max_gcr_iter = max(gcr_iters) if gcr_iters else None

    suspect = False
    reasons = []

    if max_gcr_iter is not None and max_gcr_iter >= 3000:
        suspect = True
        reasons.append(f"high_gcr_iter={max_gcr_iter}")
    if true_res is not None and true_res > 1e-7:
        suspect = True
        reasons.append(f"true_res={true_res:.3e}")

    return {
        "gcr_warning": int(gcr_warn),
        "true_residual": true_res,
        "iterated_residual": iter_res,
        "max_gcr_iter": max_gcr_iter,
        "suspect": int(suspect),
        "suspect_reason": ";".join(reasons),
    }


def safe_tag(axis: str, mm: float, suffix: str = "") -> str:
    base = f"{axis}_{mm:+.3f}mm"
    base = base.replace("+", "p").replace("-", "m").replace(".", "_")
    if suffix:
        base += f"_{suffix}"
    return base


def choose_launcher(user_choice: str) -> str:
    if user_choice == "serial":
        return "serial"
    if user_choice in ("mpirun", "mpiexec", "srun"):
        return user_choice
    if shutil.which("mpirun"):
        return "mpirun"
    if shutil.which("mpiexec"):
        return "mpiexec"
    if shutil.which("srun"):
        return "srun"
    raise RuntimeError("No encontré 'mpirun', 'mpiexec' ni 'srun' en PATH.")


def mpi_prefix(launcher: str, nprocs: int, bind: str, target: str, omp_threads: Optional[str] = None) -> list:
    if nprocs <= 1 or launcher == "serial":
        return []

    path_env = os.environ.get("PATH", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    omp_threads = omp_threads or os.environ.get("OMP_NUM_THREADS", "1")

    if launcher == "mpirun":
        prefix = [
            "mpirun",
            "-np",
            str(nprocs),
            "-genv",
            "PATH",
            path_env,
            "-genv",
            "LD_LIBRARY_PATH",
            ld_library_path,
            "-genv",
            "OMP_NUM_THREADS",
            omp_threads,
        ]
        if bind == "core":
            prefix += ["--bind-to", "core"]
        elif bind == "socket":
            prefix += ["--bind-to", "socket"]
        elif bind == "hwthread":
            prefix += ["--bind-to", "hwthread"]
        return prefix

    if launcher == "mpiexec":
        return [
            "mpiexec",
            "-n",
            str(nprocs),
            "-genv",
            "PATH",
            path_env,
            "-genv",
            "LD_LIBRARY_PATH",
            ld_library_path,
            "-genv",
            "OMP_NUM_THREADS",
            omp_threads,
        ]

    if launcher == "srun":
        prefix = ["srun", "--export=ALL", "-n", str(nprocs)]
        if bind == "core":
            prefix += ["--cpu-bind=cores"]
        elif bind == "socket":
            prefix += ["--cpu-bind=sockets"]
        elif bind == "hwthread":
            prefix += ["--cpu-bind=threads"]
        return prefix

    raise RuntimeError(f"Lanzador MPI no soportado para {target}: {launcher}")


def build_gmsh_cmd(
    launcher: str,
    mpi_procs: int,
    threads: int,
    geo_name: str,
    msh_name: str,
    bind: str,
    extra_args: list[str],
    use_time_v: bool,
) -> list:
    gmsh = shutil.which("gmsh") or "gmsh"
    base = [gmsh, geo_name, "-3", "-format", "msh2", "-save_all", "-o", msh_name]
    if threads > 0:
        base += ["-nt", str(threads)]
    if extra_args:
        base += list(extra_args)

    cmd = mpi_prefix(
        launcher=launcher,
        nprocs=mpi_procs,
        bind=bind,
        target="gmsh",
        omp_threads=str(threads) if threads > 0 else None,
    ) + base
    if use_time_v and shutil.which("/usr/bin/time"):
        return ["/usr/bin/time", "-v"] + cmd
    return cmd


def build_solver_cmd(launcher: str, nprocs: int, sif_name: str, bind: str, use_time_v: bool) -> list:
    elmer_serial = shutil.which("ElmerSolver") or "ElmerSolver"
    elmer_mpi = shutil.which("ElmerSolver_mpi") or "ElmerSolver_mpi"

    if nprocs <= 1:
        base = [elmer_serial, sif_name]
    else:
        base = mpi_prefix(launcher=launcher, nprocs=nprocs, bind=bind, target="ElmerSolver")
        base += [elmer_mpi, sif_name]

    if use_time_v and shutil.which("/usr/bin/time"):
        return ["/usr/bin/time", "-v"] + base
    return base


def partition_mesh(mesh_dir_name: str, cwd: Path, nparts: int, method: str, log_path: Path) -> None:
    if nparts <= 1:
        return

    if method == "metiskway":
        cmd = ["ElmerGrid", "2", "2", mesh_dir_name, "-metiskway", str(nparts)]
    elif method == "metisrec":
        cmd = ["ElmerGrid", "2", "2", mesh_dir_name, "-metisrec", str(nparts)]
    elif method == "simple":
        cmd = ["ElmerGrid", "2", "2", mesh_dir_name, "-partition", str(nparts), "1", "1"]
    else:
        raise RuntimeError(f"Método de partición no soportado: {method}")

    run_and_log(cmd, cwd=cwd, log_path=log_path)


def run_case(
    case_dir: Path,
    geo_src: Path,
    sif_src: Path,
    def_src: Path,
    geo_text_orig: str,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    nprocs: int,
    partition_method: str,
    launcher: str,
    bind: str,
    use_time_v: bool,
    gmsh_threads: int,
    gmsh_launcher: str,
    gmsh_mpi_procs: int,
    gmsh_extra_args: list[str],
) -> dict:
    case_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = case_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    geo_case = case_dir / geo_src.name
    sif_case = case_dir / sif_src.name
    def_case = case_dir / def_src.name

    geo_case.write_text(
        set_geo_vars(geo_text_orig, dx_m=dx_m, dy_m=dy_m, dz_m=dz_m),
        encoding="utf-8",
    )
    shutil.copy2(str(sif_src), str(sif_case))
    shutil.copy2(str(def_src), str(def_case))

    msh_name = "case.msh"
    mesh_dir_name = "mesh"

    gmsh_cmd = build_gmsh_cmd(
        launcher=gmsh_launcher,
        mpi_procs=gmsh_mpi_procs,
        threads=gmsh_threads,
        geo_name=geo_case.name,
        msh_name=msh_name,
        bind=bind,
        extra_args=gmsh_extra_args,
        use_time_v=use_time_v,
    )
    gmsh_env = os.environ.copy()
    if gmsh_threads > 0:
        gmsh_env["OMP_NUM_THREADS"] = str(gmsh_threads)
    gmsh_rc, gmsh_log_text = run_and_log_no_raise(
        gmsh_cmd,
        cwd=case_dir,
        log_path=logs_dir / "01_gmsh.log",
        env=gmsh_env,
    )
    if gmsh_rc != 0:
        tail = gmsh_log_text.splitlines()[-80:]
        raise RuntimeError(
            f"Fallo comando: {' '.join(map(str, gmsh_cmd))}\n"
            f"Ultimas lineas de {logs_dir / '01_gmsh.log'}:\n" + "\n".join(tail)
        )
    gmsh_elapsed_raw = parse_elapsed_raw(gmsh_log_text)

    run_and_log(
        ["ElmerGrid", "14", "2", msh_name, "-out", mesh_dir_name],
        cwd=case_dir,
        log_path=logs_dir / "02_elmergrid_import.log",
    )

    partition_mesh(
        mesh_dir_name=mesh_dir_name,
        cwd=case_dir,
        nparts=nprocs,
        method=partition_method,
        log_path=logs_dir / "03_elmergrid_partition.log",
    )

    solver_cmd = build_solver_cmd(
        launcher=launcher,
        nprocs=nprocs,
        sif_name=sif_case.name,
        bind=bind,
        use_time_v=use_time_v,
    )
    solver_rc, solver_log_text = run_and_log_no_raise(
        solver_cmd,
        cwd=case_dir,
        log_path=logs_dir / "04_solver.log",
        env=os.environ.copy(),
    )

    ram_gb = parse_ram_gb(solver_log_text)
    elapsed_raw = parse_elapsed_raw(solver_log_text)
    n_nodes, n_elems = parse_mesh_counts_from_msh(case_dir / msh_name)

    W = float("nan")
    if "ElectroMagnetic Field Energy" in solver_log_text:
        try:
            W = parse_energy(solver_log_text)
        except Exception:
            W = float("nan")

    diag = inspect_solver_log(solver_log_text)

    status = "OK"
    note = ""

    if math.isnan(W):
        status = "FAIL"
        note = f"solver_rc={solver_rc};no_energy_found"
    elif diag["suspect"]:
        status = "SUSPECT"
        note = diag["suspect_reason"]
    else:
        if solver_rc != 0:
            note = f"solver_rc={solver_rc};energy_present"

    # limpieza opcional para ahorrar cuota
    try:
        msh_path = case_dir / msh_name
        if msh_path.exists():
            msh_path.unlink()

        mesh_dir = case_dir / mesh_dir_name
        if mesh_dir.exists():
            shutil.rmtree(mesh_dir, ignore_errors=True)

        for pat in ("*.vtu", "*.pvtu", "*.ep", "*.result"):
            for p in case_dir.glob(pat):
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass



    return {
        "W_J": W,
        "msh_nodes": n_nodes,
        "msh_elements": n_elems,
        "case_dir": str(case_dir),
        "status": status,
        "note": note,
        "ram_gb": ram_gb,
        "elapsed_raw": elapsed_raw,
        "nprocs": nprocs,
        "launcher": launcher if nprocs > 1 else "serial",
        "partition_method": partition_method if nprocs > 1 else "none",
        "bind": bind if nprocs > 1 else "none",
        "gmsh_threads": gmsh_threads,
        "gmsh_mpi_procs": gmsh_mpi_procs,
        "gmsh_launcher": gmsh_launcher if gmsh_mpi_procs > 1 else "serial",
        "gmsh_elapsed_raw": gmsh_elapsed_raw,
        "gmsh_command": " ".join(shlex.quote(str(part)) for part in gmsh_cmd),
        **diag,
    }


def status_rank(s: str) -> int:
    return {"OK": 0, "SUSPECT": 1, "FAIL": 2, "RESCUE_OK": 0, "RESCUE_SUSPECT": 1, "RESCUE_FAIL": 2}.get(s, 9)


def pick_better_result(a: dict, b: dict) -> dict:
    if status_rank(b["status"]) < status_rank(a["status"]):
        return b
    if status_rank(b["status"]) > status_rank(a["status"]):
        return a

    # Same status: prefer smaller true residual, then fewer iters, then finite energy
    ta = a.get("true_residual") if a.get("true_residual") is not None else 1e99
    tb = b.get("true_residual") if b.get("true_residual") is not None else 1e99
    if tb < ta:
        return b
    if tb > ta:
        return a

    ia = a.get("max_gcr_iter") if a.get("max_gcr_iter") is not None else 10**9
    ib = b.get("max_gcr_iter") if b.get("max_gcr_iter") is not None else 10**9
    if ib < ia:
        return b
    return a


def to_row(res: dict, tag: str, axis: str, dx_m: float, dy_m: float, dz_m: float) -> dict:
    return {
        "tag": tag,
        "axis": axis,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dz_m": dz_m,
        "W_J": res.get("W_J"),
        "msh_nodes": res.get("msh_nodes"),
        "msh_elements": res.get("msh_elements"),
        "nprocs": res.get("nprocs"),
        "launcher": res.get("launcher"),
        "partition_method": res.get("partition_method"),
        "bind": res.get("bind"),
        "gmsh_threads": res.get("gmsh_threads"),
        "gmsh_mpi_procs": res.get("gmsh_mpi_procs"),
        "gmsh_launcher": res.get("gmsh_launcher"),
        "gmsh_elapsed_raw": res.get("gmsh_elapsed_raw"),
        "gmsh_command": res.get("gmsh_command"),
        "ram_gb": res.get("ram_gb"),
        "elapsed_raw": res.get("elapsed_raw"),
        "case_dir": res.get("case_dir"),
        "status": res.get("status"),
        "note": res.get("note"),
        "gcr_warning": res.get("gcr_warning"),
        "bad_termination": res.get("bad_termination"),
        "true_residual": res.get("true_residual"),
        "iterated_residual": res.get("iterated_residual"),
        "max_gcr_iter": res.get("max_gcr_iter"),
        "suspect": res.get("suspect"),
        "suspect_reason": res.get("suspect_reason"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Barrido 1D para Elmer con opción MPI: gmsh -> ElmerGrid -> partición -> ElmerSolver_mpi"
    )
    ap.add_argument("--geo", required=True, help="Archivo .geo de Gmsh")
    ap.add_argument("--sif", required=True, help="Archivo .sif de Elmer")
    ap.add_argument("--definition", required=True, help="Archivo circuit.definition u otro .definition")
    ap.add_argument("--outdir", default="diag_sweep_mpi", help="Directorio raíz de salida")

    ap.add_argument("--axis", choices=["dx", "dy", "dz"], default="dz", help="Eje a barrer")
    ap.add_argument("--start-mm", type=float, required=True)
    ap.add_argument("--end-mm", type=float, required=True)
    ap.add_argument("--step-mm", type=float, required=True)
    ap.add_argument("--fixed-dx-mm", type=float, default=0.0)
    ap.add_argument("--fixed-dy-mm", type=float, default=0.0)
    ap.add_argument("--fixed-dz-mm", type=float, default=0.0)

    ap.add_argument("--nprocs", type=int, default=1, help="Cantidad de procesos MPI")
    ap.add_argument(
        "--partition-method",
        choices=["metiskway", "metisrec", "simple"],
        default="metiskway",
        help="Método de partición de ElmerGrid",
    )
    ap.add_argument(
        "--launcher",
        choices=["auto", "mpirun", "mpiexec", "srun"],
        default="auto",
        help="Lanzador MPI",
    )
    ap.add_argument(
        "--bind",
        choices=["none", "core", "socket", "hwthread"],
        default="core",
        help="Afinidad CPU del launcher",
    )
    ap.add_argument("--use-time-v", action="store_true", help="Anteponer /usr/bin/time -v al solver")
    ap.add_argument(
        "--gmsh-threads",
        type=int,
        default=1,
        help="Threads OpenMP para Gmsh (-nt). Ruta recomendada para acelerar HXT.",
    )
    ap.add_argument(
        "--gmsh-launcher",
        choices=["serial", "auto", "mpirun", "mpiexec", "srun"],
        default="serial",
        help="Launcher MPI experimental para Gmsh.",
    )
    ap.add_argument(
        "--gmsh-mpi-procs",
        type=int,
        default=1,
        help="Ranks MPI para Gmsh. Experimental: Gmsh upstream indica que MPI no se usa para meshing.",
    )
    ap.add_argument(
        "--gmsh-extra-args",
        default="",
        help="Argumentos extra para Gmsh, separados como en shell. Ejemplo: '--cpu -v 4'.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Si el caso ya existe en CSV, se salta")
    ap.add_argument("--auto-rescue", action="store_true", help="Rescatar puntos FAIL o SUSPECT")
    ap.add_argument("--rescue-frac", type=float, default=0.5, help="Fracción del step para puntos de rescate")

    args = ap.parse_args()

    geo_src = Path(args.geo).resolve()
    sif_src = Path(args.sif).resolve()
    def_src = Path(args.definition).resolve()
    outdir = Path(args.outdir).resolve()
    cases_dir = outdir / "cases"
    outdir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(p) for p in [geo_src, sif_src, def_src] if not p.exists()]
    if missing:
        raise RuntimeError("No existen estos archivos de entrada:\n" + "\n".join(missing))

    if args.step_mm <= 0:
        raise RuntimeError("--step-mm debe ser > 0")
    if args.end_mm < args.start_mm:
        raise RuntimeError("--end-mm debe ser >= --start-mm")
    if args.nprocs < 1:
        raise RuntimeError("--nprocs debe ser >= 1")
    if args.gmsh_threads < 0:
        raise RuntimeError("--gmsh-threads debe ser >= 0")
    if args.gmsh_mpi_procs < 1:
        raise RuntimeError("--gmsh-mpi-procs debe ser >= 1")

    launcher = choose_launcher(args.launcher) if args.nprocs > 1 else "serial"
    gmsh_launcher = choose_launcher(args.gmsh_launcher) if args.gmsh_mpi_procs > 1 else "serial"
    gmsh_extra_args = shlex.split(args.gmsh_extra_args) if args.gmsh_extra_args.strip() else []
    geo_text_orig = geo_src.read_text(encoding="utf-8", errors="ignore")

    csv_path = outdir / "diag_sweep_results.csv"
    existing = {}
    if args.skip_existing and csv_path.exists():
        df0 = pd.read_csv(csv_path)
        if "tag" in df0.columns:
            existing = {str(t): True for t in df0["tag"].astype(str).tolist()}

    rows = []
    mm = args.start_mm
    while mm <= args.end_mm + 1e-12:
        dx_m = args.fixed_dx_mm * 1e-3
        dy_m = args.fixed_dy_mm * 1e-3
        dz_m = args.fixed_dz_mm * 1e-3

        if args.axis == "dx":
            dx_m = mm * 1e-3
        elif args.axis == "dy":
            dy_m = mm * 1e-3
        else:
            dz_m = mm * 1e-3

        tag = safe_tag(args.axis, mm)
        case_dir = cases_dir / tag

        if tag in existing:
            print(f"SKIP {tag} (ya estaba en CSV)")
            mm += args.step_mm
            continue

        try:
            res = run_case(
                case_dir=case_dir,
                geo_src=geo_src,
                sif_src=sif_src,
                def_src=def_src,
                geo_text_orig=geo_text_orig,
                dx_m=dx_m,
                dy_m=dy_m,
                dz_m=dz_m,
                nprocs=args.nprocs,
                partition_method=args.partition_method,
                launcher=launcher,
                bind=args.bind,
                use_time_v=args.use_time_v,
                gmsh_threads=args.gmsh_threads,
                gmsh_launcher=gmsh_launcher,
                gmsh_mpi_procs=args.gmsh_mpi_procs,
                gmsh_extra_args=gmsh_extra_args,
            )
        except Exception as e:
            res = {
                "W_J": float("nan"),
                "msh_nodes": None,
                "msh_elements": None,
                "case_dir": str(case_dir),
                "status": "FAIL",
                "note": f"{type(e).__name__}: {e}",
                "ram_gb": None,
                "elapsed_raw": None,
                "nprocs": args.nprocs,
                "launcher": launcher,
                "partition_method": args.partition_method if args.nprocs > 1 else "none",
                "bind": args.bind if args.nprocs > 1 else "none",
                "gmsh_threads": args.gmsh_threads,
                "gmsh_mpi_procs": args.gmsh_mpi_procs,
                "gmsh_launcher": gmsh_launcher if args.gmsh_mpi_procs > 1 else "serial",
                "gmsh_elapsed_raw": None,
                "gmsh_command": "",
                "gcr_warning": 0,
                "bad_termination": 0,
                "true_residual": None,
                "iterated_residual": None,
                "max_gcr_iter": None,
                "suspect": 0,
                "suspect_reason": "",
            }

            '''
            Quitarle el rerun a los suspect, ver después que pasa.


        # one nominal rerun if suspect
        if res["status"] == "SUSPECT":
            print(f"SUSPECT {tag}: rerun nominal")
            try:
                res2 = run_case(
                    case_dir=cases_dir / f"{tag}_rerun",
                    geo_src=geo_src,
                    sif_src=sif_src,
                    def_src=def_src,
                    geo_text_orig=geo_text_orig,
                    dx_m=dx_m,
                    dy_m=dy_m,
                    dz_m=dz_m,
                    nprocs=args.nprocs,
                    partition_method=args.partition_method,
                    launcher=launcher,
                    bind=args.bind,
                    use_time_v=args.use_time_v,
                )
                res = pick_better_result(res, res2)
            except Exception:
                pass
                
                '''

        row = to_row(res, tag, args.axis, dx_m, dy_m, dz_m)
        rows.append(row)
        print(f"{row['status']} {tag}: W={row['W_J']} | np={row['nprocs']} | {row['case_dir']}")

        # rescue around FAIL or SUSPECT
        if args.auto_rescue and row["status"] == "FAIL":
            step = args.step_mm
            alpha = args.rescue_frac
            candidates = []

            left = mm - alpha * step
            right = mm + alpha * step

            if (args.start_mm - 1e-12) <= left <= (args.end_mm + 1e-12):
                candidates.append(left)
            if (args.start_mm - 1e-12) <= right <= (args.end_mm + 1e-12):
                candidates.append(right)

            clean = []
            for c in sorted(candidates):
                if not clean or not np.isclose(c, clean[-1], rtol=0.0, atol=1e-12):
                    clean.append(c)
            candidates = clean

            if candidates:
                print(f"  -> RESCUE at {mm:+.3f} mm: running {', '.join([f'{c:+.3f}' for c in candidates])} mm")

            for mm2 in candidates:
                dx2 = args.fixed_dx_mm * 1e-3
                dy2 = args.fixed_dy_mm * 1e-3
                dz2 = args.fixed_dz_mm * 1e-3

                if args.axis == "dx":
                    dx2 = mm2 * 1e-3
                elif args.axis == "dy":
                    dy2 = mm2 * 1e-3
                else:
                    dz2 = mm2 * 1e-3

                tag2 = safe_tag(args.axis, mm2, suffix="R")

                try:
                    r2 = run_case(
                        case_dir=cases_dir / tag2,
                        geo_src=geo_src,
                        sif_src=sif_src,
                        def_src=def_src,
                        geo_text_orig=geo_text_orig,
                        dx_m=dx2,
                        dy_m=dy2,
                        dz_m=dz2,
                        nprocs=args.nprocs,
                        partition_method=args.partition_method,
                        launcher=launcher,
                        bind=args.bind,
                        use_time_v=args.use_time_v,
                        gmsh_threads=args.gmsh_threads,
                        gmsh_launcher=gmsh_launcher,
                        gmsh_mpi_procs=args.gmsh_mpi_procs,
                        gmsh_extra_args=gmsh_extra_args,
                    )
                    if r2["status"] == "OK":
                        r2["status"] = "RESCUE_OK"
                    elif r2["status"] == "SUSPECT":
                        r2["status"] = "RESCUE_SUSPECT"
                    else:
                        r2["status"] = "RESCUE_FAIL"
                    r2["note"] = (r2.get("note") or "") + f";rescued_from={tag}"
                except Exception as e:
                    r2 = {
                        "W_J": float("nan"),
                        "msh_nodes": None,
                        "msh_elements": None,
                        "case_dir": str(cases_dir / tag2),
                        "status": "RESCUE_FAIL",
                        "note": f"{type(e).__name__}: {e};rescued_from={tag}",
                        "ram_gb": None,
                        "elapsed_raw": None,
                        "nprocs": args.nprocs,
                        "launcher": launcher,
                        "partition_method": args.partition_method if args.nprocs > 1 else "none",
                        "bind": args.bind if args.nprocs > 1 else "none",
                        "gmsh_threads": args.gmsh_threads,
                        "gmsh_mpi_procs": args.gmsh_mpi_procs,
                        "gmsh_launcher": gmsh_launcher if args.gmsh_mpi_procs > 1 else "serial",
                        "gmsh_elapsed_raw": None,
                        "gmsh_command": "",
                        "gcr_warning": 0,
                        "bad_termination": 0,
                        "true_residual": None,
                        "iterated_residual": None,
                        "max_gcr_iter": None,
                        "suspect": 0,
                        "suspect_reason": "",
                    }

                rows.append(to_row(r2, tag2, args.axis, dx2, dy2, dz2))
                print(f"    {r2['status']} {tag2}: W={r2['W_J']}")

        mm += args.step_mm

    if rows:
        df = pd.DataFrame(rows)
        if csv_path.exists() and args.skip_existing:
            prev = pd.read_csv(csv_path)
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"\nWrote: {csv_path}")
    else:
        print("No hubo filas nuevas para escribir.")


if __name__ == "__main__":
    main()
