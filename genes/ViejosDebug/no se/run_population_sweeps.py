import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Corre un sweep dz por cada individuo del manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runner", required=True, help="Ruta a run_diag_sweep.py")
    ap.add_argument("--axis", default="dz", choices=["dx","dy","dz"])
    ap.add_argument("--start-mm", type=float, required=True)
    ap.add_argument("--end-mm", type=float, required=True)
    ap.add_argument("--step-mm", type=float, required=True)
    ap.add_argument("--fixed-dx-mm", type=float, default=0.0)
    ap.add_argument("--fixed-dy-mm", type=float, default=0.0)
    ap.add_argument("--fixed-dz-mm", type=float, default=0.0)
    ap.add_argument("--auto-rescue", action="store_true")
    ap.add_argument("--rescue-frac", type=float, default=0.5)
    ap.add_argument("--nominal-retries", type=int, default=0)
    ap.add_argument("--out-root", default="mini_cycle_runs")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    runner = Path(args.runner).resolve()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        ind = str(r["individual_id"])
        outdir = out_root / ind
        cmd = [
            sys.executable, str(runner),
            "--geo", str(r["geo_path"]),
            "--sif", str(r["sif_path"]),
            "--definition", str(r["definition_path"]),
            "--outdir", str(outdir),
            "--axis", args.axis,
            "--start-mm", str(args.start_mm),
            "--end-mm", str(args.end_mm),
            "--step-mm", str(args.step_mm),
            "--fixed-dx-mm", str(args.fixed_dx_mm),
            "--fixed-dy-mm", str(args.fixed_dy_mm),
            "--fixed-dz-mm", str(args.fixed_dz_mm),
            "--rescue-frac", str(args.rescue_frac),
            "--nominal-retries", str(args.nominal_retries),
        ]
        if args.auto_rescue:
            cmd.append("--auto-rescue")

        print("\n=== Running", ind, "===")
        print(" ".join(cmd))
        p = subprocess.run(cmd)
        rows.append({
            "individual_id": ind,
            "returncode": p.returncode,
            "run_dir": str(outdir),
            "diag_csv": str(outdir / "diag_sweep.csv"),
        })

    summary = pd.DataFrame(rows)
    summary_path = out_root / "run_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
