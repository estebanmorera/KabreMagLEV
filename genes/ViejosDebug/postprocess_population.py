import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Postprocesa todos los runs del mini ciclo")
    ap.add_argument("--run-summary", required=True)
    ap.add_argument("--energy-script", required=True)
    ap.add_argument("--stiffness-script", required=True)
    ap.add_argument("--coord", default="dz_m")
    ap.add_argument("--force-name", default="Fz_N")
    ap.add_argument("--smooth", default="movavg", choices=["none","movavg","savgol"])
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--poly", type=int, default=3)
    args = ap.parse_args()

    runs = pd.read_csv(args.run_summary)
    
    # normalizar columnas para distintos formatos de run_summary
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
    
    # si no existe returncode, inferimos éxito a partir de csv_exists
    if "returncode" not in runs.columns:
        if "csv_exists" in runs.columns:
            runs["returncode"] = runs["csv_exists"].map(lambda x: 0 if bool(x) else 1)
        else:
            runs["returncode"] = 0
    
    rows = []
    
    for _, r in runs.iterrows():
        ind = r["individual_id"]
        run_dir = Path(r["run_dir"])
        diag_csv = Path(r["diag_csv"])
    
        if int(r["returncode"]) != 0 or not diag_csv.exists():
            rows.append({
                "individual_id": ind,
                "run_dir": str(run_dir),
                "diag_csv": str(diag_csv),
                "post_status": "RUN_FAIL"
            })
            continue
    
        force_csv = run_dir / "energy_force_1d.csv"
        stiff_csv = run_dir / "stiffness_curve.csv"

        cmd1 = [
            sys.executable, str(Path(args.energy_script).resolve()),
            "--csv", str(diag_csv),
            "--coord", args.coord,
            "--out", str(force_csv),
            "--force-name", args.force_name,
            "--smooth", args.smooth,
            "--window", str(args.window),
            "--poly", str(args.poly),
        ]
        p1 = subprocess.run(cmd1)
        if p1.returncode != 0:
            rows.append({"individual_id": ind, "post_status": "FORCE_FAIL"})
            continue

        cmd2 = [
            sys.executable, str(Path(args.stiffness_script).resolve()),
            "--csv", str(force_csv),
            "--coord", args.coord,
            "--force", args.force_name,
            "--out", str(stiff_csv),
            "--disp-unit", "m",
        ]
        p2 = subprocess.run(cmd2)
        if p2.returncode != 0:
            rows.append({"individual_id": ind, "post_status": "STIFF_FAIL"})
            continue

        rows.append({
            "individual_id": ind,
            "post_status": "OK",
            "force_csv": str(force_csv),
            "stiffness_csv": str(stiff_csv),
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.run_summary).with_name("postprocess_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
