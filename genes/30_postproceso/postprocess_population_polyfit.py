import argparse
import subprocess
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Postprocesa toda la población usando polyfit_energy_force_stiffness_hybrid.py"
    )
    ap.add_argument("--run-summary", required=True)
    ap.add_argument("--polyfit-script", required=True)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--eval-at-mm", type=float, default=0.0)
    ap.add_argument("--window-half-mm", type=float, default=1.0)
    args = ap.parse_args()

    runs = pd.read_csv(args.run_summary)

    if "individual_id" not in runs.columns:
        if "name" in runs.columns:
            runs["individual_id"] = runs["name"]
        else:
            raise RuntimeError("El run_summary no tiene ni 'individual_id' ni 'name'")

    if "run_dir" not in runs.columns:
        if "outdir" in runs.columns:
            runs["run_dir"] = runs["outdir"]
        else:
            raise RuntimeError("El run_summary no tiene ni 'run_dir' ni 'outdir'")

    if "diag_csv" not in runs.columns:
        if "csv_path" in runs.columns:
            runs["diag_csv"] = runs["csv_path"]
        else:
            raise RuntimeError("El run_summary no tiene ni 'diag_csv' ni 'csv_path'")

    rows = []

    for _, r in runs.iterrows():
        ind = r["individual_id"]
        run_dir = Path(r["run_dir"])
        diag_csv = Path(r["diag_csv"])

        if not diag_csv.exists():
            rows.append({
                "individual_id": ind,
                "run_dir": str(run_dir),
                "diag_csv": str(diag_csv),
                "post_status": "RUN_FAIL"
            })
            continue

        out_prefix = run_dir / ind

        cmd = [
            "python3",
            str(Path(args.polyfit_script)),
            "--csv", str(diag_csv),
            "--coord", "dz_m",
            "--energy", "W_J",
            "--degree", str(args.degree),
            "--out-prefix", str(out_prefix),
            "--drop-suspect",
            "--hybrid-filter",
            "--hybrid-window", "7",
            "--hybrid-local-degree", "2",
            "--global-z", "4.5",
            "--local-z", "4.5",
            "--jump-z", "4.5",
            "--branch-z", "4.0",
            "--hybrid-iters", "2",
            "--robust-fit", "huber",
            "--stiffness-sign", "paper",
            "--eval-at-mm", str(args.eval_at_mm),
        ]

        p = subprocess.run(cmd, text=True, capture_output=True)

        fit_curve_csv = Path(str(out_prefix) + ".fit_curve.csv")
        fit_points_csv = Path(str(out_prefix) + ".fit_points.csv")
        fit_summary_txt = Path(str(out_prefix) + ".summary.txt")
        fit_png = Path(str(out_prefix) + ".png")

        rows.append({
            "individual_id": ind,
            "run_dir": str(run_dir),
            "diag_csv": str(diag_csv),
            "fit_curve_csv": str(fit_curve_csv),
            "fit_points_csv": str(fit_points_csv),
            "fit_summary_txt": str(fit_summary_txt),
            "fit_png": str(fit_png),
            "post_status": "OK" if p.returncode == 0 and fit_curve_csv.exists() else "POST_FAIL",
            "returncode": p.returncode,
            "stdout_tail": p.stdout[-1500:],
            "stderr_tail": p.stderr[-1500:],
        })

    out_csv = Path(args.run_summary).with_name("postprocess_polyfit_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()