#!/usr/bin/env python3
"""Run one manifest row as a standalone Slurm array task.

Recommended usage from an sbatch array job:

    python3 genes_slurm_eval_individual.py \
      --manifest /work/jmorera/Genes/testparallel_slurm/RUN/results/01_population/population_manifest.csv \
      --runner /work/jmorera/Genes/genes/20_ejecucion/sweep_mpi.py \
      --project-cwd /work/jmorera/Genes \
      --out-root /work/jmorera/Genes/testparallel_slurm/RUN/runs \
      --index "$SLURM_ARRAY_TASK_ID" \
      --nprocs "$SLURM_NTASKS" \
      --launcher srun \
      --axis dz \
      --start-mm -3.0 \
      --end-mm 3.0 \
      --step-mm 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


OK_STATUSES = {"OK", "RESCUE_OK"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV con population_manifest.")
    parser.add_argument("--runner", required=True, help="Ruta a sweep_mpi.py.")
    parser.add_argument("--project-cwd", required=True, help="Working directory del proyecto Genes.")
    parser.add_argument("--out-root", required=True, help="Raiz donde se guardan runs/<individual_id>.")
    parser.add_argument("--index", type=int, default=None, help="Indice 0-based dentro del manifest.")
    parser.add_argument("--individual-id", default="", help="Alternativa a --index.")
    parser.add_argument("--geo-name", default="StepsHTX.geo")
    parser.add_argument("--sif-name", default="P1low.sif")
    parser.add_argument("--definition-name", default="HMB_circuit.definition")
    parser.add_argument("--axis", default="dz")
    parser.add_argument("--fixed-dx-mm", type=float, default=0.0)
    parser.add_argument("--fixed-dy-mm", type=float, default=0.0)
    parser.add_argument("--fixed-dz-mm", type=float, default=0.0)
    parser.add_argument("--start-mm", type=float, required=True)
    parser.add_argument("--end-mm", type=float, required=True)
    parser.add_argument("--step-mm", type=float, required=True)
    parser.add_argument("--nprocs", type=int, default=0)
    parser.add_argument("--partition-method", default="metiskway")
    parser.add_argument("--launcher", default="srun")
    parser.add_argument("--bind", default="core", help="Deja vacio para no pasar --bind al runner.")
    parser.add_argument("--auto-rescue", action="store_true")
    parser.add_argument("--rescue-frac", type=float, default=0.5)
    parser.add_argument("--python-exe", default="python3")
    parser.add_argument("--timeout-sec", type=int, default=0, help="0 desactiva timeout.")
    parser.add_argument("--label", default="", help="Etiqueta humana del experimento.")
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} no contiene filas.")
    if "individual_id" not in rows[0]:
        raise ValueError(f"{path} no contiene columna individual_id.")
    return rows


def select_row(rows: list[dict[str, str]], index: int | None, individual_id: str) -> tuple[int, dict[str, str]]:
    if individual_id:
        for i, row in enumerate(rows):
            if str(row.get("individual_id", "")) == individual_id:
                return i, row
        raise ValueError(f"No encontre individual_id={individual_id} en el manifest.")

    if index is None:
        env_index = os.getenv("SLURM_ARRAY_TASK_ID", "")
        if env_index == "":
            raise ValueError("Debes pasar --index o definir SLURM_ARRAY_TASK_ID.")
        index = int(env_index)

    if index < 0 or index >= len(rows):
        raise IndexError(f"Indice {index} fuera de rango para {len(rows)} filas.")
    return index, rows[index]


def find_case_file(row: dict[str, str], key: str, case_dir: Path, fallback_name: str) -> Path:
    raw = str(row.get(key, "")).strip()
    if raw:
        return Path(raw)
    return case_dir / fallback_name


def find_result_csv(run_dir: Path) -> Path | None:
    for name in ("diag_sweep_results.csv", "diag_sweep.csv"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    return None


def summarize_result_csv(csv_path: Path | None) -> tuple[int | None, int | None, int | None]:
    if csv_path is None or not csv_path.exists():
        return None, None, None

    rows = 0
    ok_rows = 0
    fail_rows = 0
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        status_key = "status" if reader.fieldnames and "status" in reader.fieldnames else ""
        for row in reader:
            rows += 1
            if status_key:
                status = str(row.get(status_key, "")).strip().upper()
                if status in OK_STATUSES:
                    ok_rows += 1
                else:
                    fail_rows += 1

    if rows and ok_rows == 0 and fail_rows == 0:
        ok_rows = None
        fail_rows = None
    return rows, ok_rows, fail_rows


def build_runner_command(args: argparse.Namespace, row: dict[str, str], run_dir: Path) -> list[str]:
    case_dir = Path(str(row["case_dir"]))
    geo = find_case_file(row, "geo_path", case_dir, args.geo_name)
    sif = find_case_file(row, "sif_path", case_dir, args.sif_name)
    definition = find_case_file(row, "definition_path", case_dir, args.definition_name)
    nprocs = args.nprocs or int(os.getenv("SLURM_NTASKS", "0") or "0")
    if nprocs <= 0:
        raise ValueError("No pude resolver nprocs; pasa --nprocs o define SLURM_NTASKS.")

    cmd = [
        args.python_exe,
        str(Path(args.runner)),
        "--geo",
        str(geo),
        "--sif",
        str(sif),
        "--definition",
        str(definition),
        "--outdir",
        str(run_dir),
        "--axis",
        args.axis,
        "--fixed-dx-mm",
        str(args.fixed_dx_mm),
        "--fixed-dy-mm",
        str(args.fixed_dy_mm),
        "--fixed-dz-mm",
        str(args.fixed_dz_mm),
        "--start-mm",
        str(args.start_mm),
        "--end-mm",
        str(args.end_mm),
        "--step-mm",
        str(args.step_mm),
        "--nprocs",
        str(nprocs),
        "--partition-method",
        args.partition_method,
        "--launcher",
        args.launcher,
    ]
    if args.bind.strip():
        cmd.extend(["--bind", args.bind.strip()])
    if args.auto_rescue:
        cmd.extend(["--auto-rescue", "--rescue-frac", str(args.rescue_frac)])
    return cmd


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    project_cwd = Path(args.project_cwd).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest)
    index, row = select_row(rows, args.index, args.individual_id)
    individual_id = str(row["individual_id"])
    run_dir = out_root / individual_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    metadata_path = run_dir / "run_metadata.json"

    cmd = build_runner_command(args, row, run_dir)

    metadata = {
        "label": args.label,
        "manifest": str(manifest),
        "manifest_index": index,
        "individual_id": individual_id,
        "case_dir": row.get("case_dir", ""),
        "run_dir": str(run_dir),
        "project_cwd": str(project_cwd),
        "command": " ".join(shlex.quote(part) for part in cmd),
        "nprocs": args.nprocs or int(os.getenv("SLURM_NTASKS", "0") or "0"),
        "launcher": args.launcher,
        "bind": args.bind,
        "auto_rescue": args.auto_rescue,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_step_id": os.getenv("SLURM_STEP_ID", ""),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID", ""),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID", ""),
        "slurm_nodeid": os.getenv("SLURM_NODEID", ""),
        "slurm_localid": os.getenv("SLURM_LOCALID", ""),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    start = time.time()
    returncode = None
    timed_out = False
    env = os.environ.copy()
    # When the notebook itself runs inside a Slurm allocation, inner launchers can inherit
    # conflicting scheduler variables from the parent session.
    #
    # - For inner srun: strip conflicting memory knobs from the parent allocation.
    # - For inner mpirun/mpiexec: also strip Slurm/PMI variables so the MPI launcher
    #   treats this as a plain local launch on the allocated node instead of assuming
    #   the outer Slurm step has only one task available.
    for key in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_MEM_PER_NODE"):
        env.pop(key, None)
    if args.launcher.strip().lower() in {"mpirun", "mpiexec", "mpiexec.hydra"}:
        for key in list(env.keys()):
            if key.startswith("SLURM_") or key.startswith("PMI_") or key.startswith("PMIX_"):
                env.pop(key, None)

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(project_cwd),
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=env,
                timeout=args.timeout_sec if args.timeout_sec > 0 else None,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -999

    elapsed = time.time() - start
    csv_path = find_result_csv(run_dir)
    rows_count, ok_rows, fail_rows = summarize_result_csv(csv_path)

    metadata.update(
        {
            "started_at_epoch": start,
            "finished_at_epoch": time.time(),
            "seconds": elapsed,
            "returncode": returncode,
            "timeout": timed_out,
            "csv_path": str(csv_path) if csv_path else "",
            "rows": rows_count,
            "ok_rows": ok_rows,
            "fail_rows": fail_rows,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metadata, indent=2))
    return 0 if returncode == 0 else int(returncode)


if __name__ == "__main__":
    sys.exit(main())
