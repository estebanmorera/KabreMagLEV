#!/usr/bin/env python3
"""Collect per-individual run outputs into one CSV summary.

This script is meant to run after a Slurm array finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


OK_STATUSES = {"OK", "RESCUE_OK"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV con population_manifest.")
    parser.add_argument("--runs-root", required=True, help="Raiz donde viven runs/<individual_id>.")
    parser.add_argument("--out", required=True, help="CSV final run_summary.csv.")
    parser.add_argument("--label", default="")
    parser.add_argument("--expected-rows", type=int, default=0, help="Filas esperadas por individuo.")
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} no contiene filas.")
    return rows


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


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_rows(manifest)
    summary_rows: list[dict] = []

    for row in rows:
        individual_id = str(row["individual_id"])
        run_dir = runs_root / individual_id
        csv_path = find_result_csv(run_dir)
        rows_count, ok_rows, fail_rows = summarize_result_csv(csv_path)
        metadata = load_metadata(run_dir)

        complete = bool(csv_path and csv_path.exists())
        if args.expected_rows > 0:
            complete = complete and rows_count == args.expected_rows

        summary_rows.append(
            {
                "label": args.label,
                "name": individual_id,
                "individual_id": individual_id,
                "case_dir": row.get("case_dir", ""),
                "outdir": str(run_dir),
                "csv_path": str(csv_path) if csv_path else "",
                "csv_exists": bool(csv_path),
                "rows": rows_count,
                "ok_rows": ok_rows,
                "fail_rows": fail_rows,
                "seconds": metadata.get("seconds"),
                "returncode": metadata.get("returncode"),
                "timeout": metadata.get("timeout"),
                "nprocs": metadata.get("nprocs"),
                "launcher": metadata.get("launcher"),
                "bind": metadata.get("bind"),
                "auto_rescue": metadata.get("auto_rescue"),
                "hostname": metadata.get("hostname"),
                "slurm_job_id": metadata.get("slurm_job_id"),
                "slurm_step_id": metadata.get("slurm_step_id"),
                "slurm_array_job_id": metadata.get("slurm_array_job_id"),
                "slurm_array_task_id": metadata.get("slurm_array_task_id"),
                "slurm_nodeid": metadata.get("slurm_nodeid"),
                "slurm_localid": metadata.get("slurm_localid"),
                "complete": complete,
            }
        )

    fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "label",
        "name",
        "individual_id",
        "case_dir",
        "outdir",
        "csv_path",
        "csv_exists",
        "rows",
        "ok_rows",
        "fail_rows",
        "seconds",
        "returncode",
        "timeout",
        "nprocs",
        "launcher",
        "bind",
        "auto_rescue",
        "hostname",
        "slurm_job_id",
        "slurm_step_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "slurm_nodeid",
        "slurm_localid",
        "complete",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    complete_rows = sum(1 for row in summary_rows if row["complete"])
    print(f"Summary written to {out}")
    print(f"Complete runs: {complete_rows}/{len(summary_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
