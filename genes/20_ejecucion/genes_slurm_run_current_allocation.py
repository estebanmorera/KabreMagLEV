#!/usr/bin/env python3
"""Dispatch individuals across nodes in the current Slurm allocation.

The intended usage is:
1. Jupyter (already running inside a multi-node allocation) prepares the manifest/cases.
2. This script detects the nodes assigned to the current job.
3. One worker thread per node launches an outer `srun` step pinned to that node.
4. The per-individual helper (`genes_slurm_eval_individual.py`) runs on that node and
   launches the actual MPI simulation with `mpirun` or the configured inner launcher.

This avoids nested `sbatch` and keeps the pattern "one simulation per node".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue


OK_STATUSES = {"OK", "RESCUE_OK"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-helper", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--project-cwd", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--dispatch-root", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--allocation-info-out", default="")
    parser.add_argument("--module-preamble-file", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--axis", default="dz")
    parser.add_argument("--fixed-dx-mm", type=float, default=0.0)
    parser.add_argument("--fixed-dy-mm", type=float, default=0.0)
    parser.add_argument("--fixed-dz-mm", type=float, default=0.0)
    parser.add_argument("--start-mm", type=float, required=True)
    parser.add_argument("--end-mm", type=float, required=True)
    parser.add_argument("--step-mm", type=float, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--nprocs", type=int, required=True)
    parser.add_argument("--partition-method", default="metiskway")
    parser.add_argument("--inner-launcher", default="mpirun")
    parser.add_argument("--bind", default="")
    parser.add_argument("--gmsh-threads", type=int, default=1)
    parser.add_argument("--gmsh-launcher", default="serial")
    parser.add_argument("--gmsh-mpi-procs", type=int, default=1)
    parser.add_argument("--gmsh-extra-args", default="")
    parser.add_argument("--auto-rescue", action="store_true")
    parser.add_argument("--rescue-frac", type=float, default=0.5)
    parser.add_argument("--python-exe-remote", default="python3")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-individuals", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=0)
    parser.add_argument("--skip-first-node", action="store_true")
    parser.add_argument("--only-individual-id", action="append", default=[])
    parser.add_argument("--node", action="append", default=[], help="Manual node override; can be repeated.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} no contiene filas.")
    if "individual_id" not in rows[0]:
        raise ValueError(f"{path} no contiene columna individual_id.")
    return rows


def capture(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def detect_nodes(args: argparse.Namespace, project_cwd: Path) -> dict:
    job_id = os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOBID") or ""
    nodelist_env = os.getenv("SLURM_JOB_NODELIST") or os.getenv("SLURM_NODELIST") or ""
    nodes: list[str] = []
    error = ""

    if args.node:
        nodes = [str(node).strip() for node in args.node if str(node).strip()]
    elif nodelist_env:
        rc, stdout, stderr = capture(["bash", "-lc", f"scontrol show hostnames '{nodelist_env}'"], cwd=project_cwd)
        if rc == 0:
            nodes = [line.strip() for line in stdout.splitlines() if line.strip()]
        else:
            error = stderr.strip() or stdout.strip()

    all_nodes = list(nodes)
    if args.skip_first_node and len(nodes) > 1:
        nodes = nodes[1:]
    if args.max_nodes and args.max_nodes > 0:
        nodes = nodes[: args.max_nodes]

    return {
        "job_id": job_id,
        "nodelist_env": nodelist_env,
        "all_nodes": all_nodes,
        "active_nodes": nodes,
        "slurm_job_num_nodes": os.getenv("SLURM_JOB_NUM_NODES", ""),
        "slurm_ntasks": os.getenv("SLURM_NTASKS", ""),
        "hostname": os.getenv("HOSTNAME", ""),
        "error": error,
    }


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
        has_status = bool(reader.fieldnames and "status" in reader.fieldnames)
        for row in reader:
            rows += 1
            if has_status:
                status = str(row.get("status", "")).strip().upper()
                if status in OK_STATUSES:
                    ok_rows += 1
                else:
                    fail_rows += 1

    if rows and ok_rows == 0 and fail_rows == 0:
        ok_rows = None
        fail_rows = None
    return rows, ok_rows, fail_rows


def load_metadata(run_dir: Path) -> dict:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_complete(individual_id: str, out_root: Path, expected_rows: int) -> bool:
    run_dir = out_root / individual_id
    csv_path = find_result_csv(run_dir)
    rows, _, _ = summarize_result_csv(csv_path)
    if not csv_path or rows != expected_rows:
        return False
    metadata = load_metadata(run_dir)
    if metadata and metadata.get("returncode") not in (0, None):
        return False
    return True


def choose_todo_ids(rows: list[dict[str, str]], args: argparse.Namespace, out_root: Path) -> list[str]:
    ids = [str(row["individual_id"]) for row in rows]
    if args.only_individual_id:
        wanted = {str(x) for x in args.only_individual_id}
        ids = [individual_id for individual_id in ids if individual_id in wanted]
    if args.max_individuals and args.max_individuals > 0:
        ids = ids[: args.max_individuals]
    if args.skip_completed:
        ids = [individual_id for individual_id in ids if not is_complete(individual_id, out_root, args.expected_rows)]
    return ids


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = [
            "individual_id",
            "assigned_node",
            "outer_returncode",
            "outer_seconds",
            "helper_returncode",
            "helper_seconds",
            "csv_exists",
            "rows",
            "ok_rows",
            "fail_rows",
            "hostname",
            "slurm_job_id",
            "slurm_step_id",
            "complete",
            "mode",
        ]
    else:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part) != "")


def build_helper_parts(args: argparse.Namespace, individual_id: str, manifest: Path, eval_helper: Path, runner: Path, project_cwd: Path, out_root: Path) -> list[str]:
    parts = [
        args.python_exe_remote,
        str(eval_helper),
        "--manifest",
        str(manifest),
        "--runner",
        str(runner),
        "--project-cwd",
        str(project_cwd),
        "--out-root",
        str(out_root),
        "--individual-id",
        individual_id,
        "--nprocs",
        str(args.nprocs),
        "--launcher",
        args.inner_launcher,
        "--partition-method",
        args.partition_method,
        "--gmsh-threads",
        str(args.gmsh_threads),
        "--gmsh-launcher",
        args.gmsh_launcher,
        "--gmsh-mpi-procs",
        str(args.gmsh_mpi_procs),
        "--axis",
        args.axis,
        "--start-mm",
        str(args.start_mm),
        "--end-mm",
        str(args.end_mm),
        "--step-mm",
        str(args.step_mm),
        "--fixed-dx-mm",
        str(args.fixed_dx_mm),
        "--fixed-dy-mm",
        str(args.fixed_dy_mm),
        "--fixed-dz-mm",
        str(args.fixed_dz_mm),
        "--label",
        args.label,
    ]
    if args.bind.strip():
        parts.extend(["--bind", args.bind.strip()])
    if args.gmsh_extra_args.strip():
        parts.extend(["--gmsh-extra-args", args.gmsh_extra_args.strip()])
    if args.auto_rescue:
        parts.extend(["--auto-rescue", "--rescue-frac", str(args.rescue_frac)])
    if args.timeout_sec and args.timeout_sec > 0:
        parts.extend(["--timeout-sec", str(args.timeout_sec)])
    return parts


def run_one_individual(
    node_name: str,
    individual_id: str,
    args: argparse.Namespace,
    manifest: Path,
    eval_helper: Path,
    runner: Path,
    project_cwd: Path,
    out_root: Path,
    dispatch_root: Path,
    module_preamble: str,
) -> dict:
    dispatch_dir = dispatch_root / individual_id
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    remote_script_path = dispatch_dir / "remote_step.sh"
    command_path = dispatch_dir / "outer_step_command.txt"
    stdout_path = dispatch_dir / "outer_stdout.log"
    stderr_path = dispatch_dir / "outer_stderr.log"

    helper_parts = build_helper_parts(args, individual_id, manifest, eval_helper, runner, project_cwd, out_root)
    modules_bootstrap = "\n".join(
        [
            "if ! command -v module >/dev/null 2>&1; then",
            "  for init_file in \\",
            "    /etc/profile.d/modules.sh \\",
            "    /etc/profile.d/lmod.sh \\",
            "    /usr/share/Modules/init/bash \\",
            "    /usr/share/modules/init/bash \\",
            "    /opt/ohpc/admin/lmod/lmod/init/bash; do",
            "    if [ -f \"$init_file\" ]; then",
            "      . \"$init_file\"",
            "      break",
            "    fi",
            "  done",
            "fi",
            "if ! command -v module >/dev/null 2>&1; then",
            "  echo 'module command not available after bootstrap' >&2",
            "  exit 127",
            "fi",
        ]
    )
    
    remote_script = "\n".join(
        [
            "#!/bin/bash",
            "set -eo pipefail",
            f"cd {project_cwd}",
            modules_bootstrap,
            module_preamble.rstrip(),
            "unset SLURM_MEM_PER_CPU",
            "unset SLURM_MEM_PER_GPU",
            "unset SLURM_MEM_PER_NODE",
            shell_join(helper_parts),
            "",
        ]
    )
    remote_script_path.write_text(remote_script, encoding="utf-8")

    outer_parts = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={args.nprocs}",
        "--cpu-bind=none",
        "--exact",
        "--exclusive",
        f"--nodelist={node_name}",
        "/bin/bash",
        "-lc",
        f". {shlex.quote(str(remote_script_path))}",
    ]
    command_path.write_text(" ".join(outer_parts) + "\n", encoding="utf-8")

    start = time.time()
    if args.dry_run:
        return {
            "individual_id": individual_id,
            "assigned_node": node_name,
            "outer_returncode": None,
            "outer_seconds": 0.0,
            "helper_returncode": None,
            "helper_seconds": None,
            "csv_exists": False,
            "rows": None,
            "ok_rows": None,
            "fail_rows": None,
            "hostname": "",
            "slurm_job_id": "",
            "slurm_step_id": "",
            "complete": False,
            "mode": "DRY_RUN",
        }

    proc = subprocess.run(
        outer_parts,
        cwd=str(project_cwd),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    run_dir = out_root / individual_id
    csv_path = find_result_csv(run_dir)
    rows, ok_rows, fail_rows = summarize_result_csv(csv_path)
    metadata = load_metadata(run_dir)
    helper_returncode = metadata.get("returncode")
    complete = bool(csv_path and rows == args.expected_rows and helper_returncode == 0)

    return {
        "individual_id": individual_id,
        "assigned_node": node_name,
        "outer_returncode": proc.returncode,
        "outer_seconds": round(elapsed, 3),
        "helper_returncode": helper_returncode,
        "helper_seconds": metadata.get("seconds"),
        "csv_exists": bool(csv_path),
        "rows": rows,
        "ok_rows": ok_rows,
        "fail_rows": fail_rows,
        "hostname": metadata.get("hostname", ""),
        "slurm_job_id": metadata.get("slurm_job_id", ""),
        "slurm_step_id": metadata.get("slurm_step_id", ""),
        "complete": complete,
        "mode": "executed",
    }


def main() -> int:
    args = parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    eval_helper = Path(args.eval_helper).expanduser().resolve()
    runner = Path(args.runner).expanduser().resolve()
    project_cwd = Path(args.project_cwd).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    dispatch_root = Path(args.dispatch_root).expanduser().resolve()
    summary_out = Path(args.summary_out).expanduser().resolve()
    allocation_info_out = Path(args.allocation_info_out).expanduser().resolve() if args.allocation_info_out else None
    module_preamble_file = Path(args.module_preamble_file).expanduser().resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    dispatch_root.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest)
    todo_ids = choose_todo_ids(rows, args, out_root)
    allocation_info = detect_nodes(args, project_cwd)
    allocation_info["todo_ids"] = todo_ids
    allocation_info["expected_rows"] = args.expected_rows
    allocation_info["nprocs_per_simulation"] = args.nprocs
    allocation_info["inner_launcher"] = args.inner_launcher
    allocation_info["gmsh_threads"] = args.gmsh_threads
    allocation_info["gmsh_launcher"] = args.gmsh_launcher
    allocation_info["gmsh_mpi_procs"] = args.gmsh_mpi_procs
    allocation_info["gmsh_extra_args"] = args.gmsh_extra_args

    if allocation_info_out:
        write_json(allocation_info_out, allocation_info)

    print(json.dumps(allocation_info, indent=2))

    active_nodes = allocation_info["active_nodes"]
    if not active_nodes:
        print("No pude resolver nodos activos.", file=sys.stderr)
        return 2

    if not todo_ids:
        print("No hay individuos pendientes.")
        write_csv(summary_out, [])
        return 0

    module_preamble = module_preamble_file.read_text(encoding="utf-8")

    if args.dry_run:
        preview = []
        for i, individual_id in enumerate(todo_ids[: len(active_nodes)]):
            preview.append(
                {
                    "individual_id": individual_id,
                    "assigned_node": active_nodes[i % len(active_nodes)],
                }
            )
        write_csv(summary_out, preview)
        print("Dry run preview written to", summary_out)
        return 0

    todo_queue: Queue[str] = Queue()
    for individual_id in todo_ids:
        todo_queue.put(individual_id)

    results: list[dict] = []
    lock = threading.Lock()
    stop_event = threading.Event()

    def worker(node_name: str) -> list[dict]:
        local_results: list[dict] = []
        while not stop_event.is_set():
            try:
                individual_id = todo_queue.get_nowait()
            except Empty:
                break

            result = run_one_individual(
                node_name=node_name,
                individual_id=individual_id,
                args=args,
                manifest=manifest,
                eval_helper=eval_helper,
                runner=runner,
                project_cwd=project_cwd,
                out_root=out_root,
                dispatch_root=dispatch_root,
                module_preamble=module_preamble,
            )
            local_results.append(result)

            with lock:
                results.append(result)
                write_csv(summary_out, results)
                print(
                    f"[{node_name}] {individual_id} | outer_rc={result['outer_returncode']} | "
                    f"helper_rc={result['helper_returncode']} | rows={result['rows']} | complete={result['complete']}"
                )
                if args.fail_fast and not result["complete"]:
                    stop_event.set()

            todo_queue.task_done()
        return local_results

    with ThreadPoolExecutor(max_workers=len(active_nodes)) as ex:
        futures = {ex.submit(worker, node_name): node_name for node_name in active_nodes}
        for fut in as_completed(futures):
            node_name = futures[fut]
            fut.result()
            print(f"Worker terminado en {node_name}")

    has_failures = any(not row.get("complete") for row in results)
    return 2 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
