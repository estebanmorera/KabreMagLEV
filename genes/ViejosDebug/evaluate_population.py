import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_zero_crossing_width(x_mm, k_n_per_mm):
    x = np.asarray(x_mm, float)
    k = np.asarray(k_n_per_mm, float)
    order = np.argsort(x)
    x = x[order]
    k = k[order]

    def interp_cross(i):
        x0, x1 = x[i], x[i+1]
        y0, y1 = k[i], k[i+1]
        if y1 == y0:
            return 0.5 * (x0 + x1)
        return x0 - y0 * (x1 - x0) / (y1 - y0)

    left = None
    right = None
    center_idx = np.argmin(np.abs(x))

    for i in range(center_idx - 1, -1, -1):
        if k[i] == 0:
            left = x[i]
            break
        if k[i] * k[i+1] < 0:
            left = interp_cross(i)
            break

    for i in range(center_idx, len(x) - 1):
        if k[i] == 0:
            right = x[i]
            break
        if k[i] * k[i+1] < 0:
            right = interp_cross(i)
            break

    if left is None:
        left = float(x[0])
    if right is None:
        right = float(x[-1])

    return float(left), float(right), float(right - left)


def compute_metrics(stiff_df: pd.DataFrame, force_df: pd.DataFrame, window_half_mm: float):
    sdf = stiff_df.sort_values("z_mm").reset_index(drop=True)
    fdf = force_df.sort_values("z_mm").reset_index(drop=True)

    # z_eq desde cruce F=0 más cercano a cero
    z = fdf["z_mm"].to_numpy(float)
    F = fdf["Fz_N"].to_numpy(float)
    idx = np.argmin(np.abs(z))
    z_eq = float(z[idx])
    F_eq = float(F[idx])
    for i in range(len(z) - 1):
        if F[i] == 0.0:
            cand = z[i]
        elif F[i] * F[i+1] < 0:
            cand = z[i] - F[i] * (z[i+1] - z[i]) / (F[i+1] - F[i])
        else:
            continue
        if abs(cand) < abs(z_eq):
            z_eq = float(cand)
            F_eq = 0.0

    kx = sdf["z_mm"].to_numpy(float)
    kk = sdf["k_N_per_mm"].to_numpy(float)
    # K_eq interpolado simple
    K_eq = float(np.interp(z_eq, kx, kk))

    stable_left, stable_right, stable_width = find_zero_crossing_width(kx, kk)

    mask = (kx >= z_eq - window_half_mm) & (kx <= z_eq + window_half_mm)
    win = sdf.loc[mask].copy()
    if len(win) == 0:
        raise RuntimeError("No hay puntos dentro de la ventana de evaluación")

    return {
        "n_points": len(fdf),
        "z_eq_mm": z_eq,
        "F_eq_N": F_eq,
        "K_eq_N_per_mm": K_eq,
        "stable_left_mm": stable_left,
        "stable_right_mm": stable_right,
        "stable_width_mm": stable_width,
        "K_min_window_N_per_mm": float(win["k_N_per_mm"].min()),
        "K_mean_window_N_per_mm": float(win["k_N_per_mm"].mean()),
        "K_max_window_N_per_mm": float(win["k_N_per_mm"].max()),
        "window_half_mm": window_half_mm,
        "window_points": int(len(win)),
    }


def minmax_benefit(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def minmax_cost(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (hi - s) / (hi - lo)


def main():
    ap = argparse.ArgumentParser(description="Evalúa el mini ciclo a partir de curvas de fuerza y rigidez")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run-summary", required=True)
    ap.add_argument("--window-half-mm", type=float, default=1.0)
    ap.add_argument("--out", default="population_evaluation.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    runs = pd.read_csv(args.run_summary)
    rows = []

    for _, rr in runs.iterrows():
        ind = rr["individual_id"]
        if int(rr["returncode"]) != 0:
            rows.append({"individual_id": ind, "status_eval": "RUN_FAIL"})
            continue

        run_dir = Path(rr["run_dir"])
        force_csv = run_dir / "energy_force_1d.csv"
        stiff_csv = run_dir / "stiffness_curve.csv"
        if not force_csv.exists() or not stiff_csv.exists():
            rows.append({"individual_id": ind, "status_eval": "MISSING_POST"})
            continue

        force_df = pd.read_csv(force_csv)
        stiff_df = pd.read_csv(stiff_csv)
        # normalizar nombres esperados
        if "dz_m" not in force_df.columns or "Fz_N" not in force_df.columns:
            raise RuntimeError(f"{force_csv} no tiene columnas dz_m/Fz_N esperadas")
        if "dz_m" not in stiff_df.columns or "k_N_per_mm" not in stiff_df.columns:
            raise RuntimeError(f"{stiff_csv} no tiene columnas dz_m/k_N_per_mm esperadas")
        force_df = force_df.rename(columns={"dz_m": "z_m"})
        stiff_df = stiff_df.rename(columns={"dz_m": "z_m"})
        force_df["z_mm"] = force_df["z_m"] * 1e3
        stiff_df["z_mm"] = stiff_df["z_m"] * 1e3

        met = compute_metrics(stiff_df, force_df, args.window_half_mm)
        rows.append({"individual_id": ind, "status_eval": "OK", **met})

    out = pd.DataFrame(rows).merge(manifest, on="individual_id", how="left")
    ok = out["status_eval"] == "OK"
    if ok.any():
        out.loc[ok, "score_range"] = minmax_benefit(out.loc[ok, "stable_width_mm"])
        out.loc[ok, "score_stiffness"] = minmax_benefit(0.7 * out.loc[ok, "K_mean_window_N_per_mm"] + 0.3 * out.loc[ok, "K_min_window_N_per_mm"])
        out.loc[ok, "score_current"] = minmax_cost(out.loc[ok, "I_eval_A"].abs())
        out.loc[ok, "score_zeq"] = minmax_cost(out.loc[ok, "z_eq_mm"].abs())
        out.loc[ok, "score_total"] = (
            0.35 * out.loc[ok, "score_range"] +
            0.40 * out.loc[ok, "score_stiffness"] +
            0.20 * out.loc[ok, "score_current"] +
            0.05 * out.loc[ok, "score_zeq"]
        )
    out = out.sort_values(["status_eval", "score_total"], ascending=[True, False])
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    cols = [c for c in [
        "individual_id","family","status_eval","stable_width_mm",
        "K_mean_window_N_per_mm","K_min_window_N_per_mm","z_eq_mm",
        "I_eval_A","score_total"
    ] if c in out.columns]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
