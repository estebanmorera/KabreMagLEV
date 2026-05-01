import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def interp_at(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    return float(np.interp(xq, x, y))


def roots_from_samples(x: np.ndarray, y: np.ndarray):
    roots = []
    for i in range(len(x) - 1):
        x1, x2 = float(x[i]), float(x[i + 1])
        y1, y2 = float(y[i]), float(y[i + 1])

        if np.isclose(y1, 0.0):
            roots.append(x1)

        if y1 == 0.0 and y2 == 0.0:
            continue

        if y1 * y2 < 0.0:
            xr = x1 - y1 * (x2 - x1) / (y2 - y1)
            roots.append(float(xr))

    if np.isclose(y[-1], 0.0):
        roots.append(float(x[-1]))

    out = []
    for r in sorted(roots):
        if not out or not np.isclose(r, out[-1], rtol=0.0, atol=1e-9):
            out.append(r)
    return out


def choose_equilibrium_root(x: np.ndarray, f: np.ndarray) -> float:
    roots = roots_from_samples(x, f)
    if roots:
        return min(roots, key=lambda r: abs(r))
    return float(x[np.argmin(np.abs(f))])


def positive_interval_around_reference(x: np.ndarray, k: np.ndarray, xref: float):
    n = len(x)
    if n < 2:
        return None, None, 0.0

    kref = interp_at(x, k, xref)
    if kref <= 0.0:
        return None, None, 0.0

    idx = np.searchsorted(x, xref) - 1
    idx = max(0, min(idx, n - 2))

    left = xref
    i = idx
    while i >= 0:
        x1, x2 = float(x[i]), float(x[i + 1])
        k1, k2 = float(k[i]), float(k[i + 1])

        if k1 > 0.0 and k2 > 0.0:
            left = x1
            i -= 1
            continue

        if k1 * k2 < 0.0:
            xr = x1 - k1 * (x2 - x1) / (k2 - k1)
            left = float(xr)
            break

        if np.isclose(k1, 0.0):
            left = x1
            break
        if np.isclose(k2, 0.0):
            left = x2
            break
        break

    right = xref
    i = idx
    while i < n - 1:
        x1, x2 = float(x[i]), float(x[i + 1])
        k1, k2 = float(k[i]), float(k[i + 1])

        if k1 > 0.0 and k2 > 0.0:
            right = x2
            i += 1
            continue

        if k1 * k2 < 0.0:
            xr = x1 - k1 * (x2 - x1) / (k2 - k1)
            right = float(xr)
            break

        if np.isclose(k1, 0.0):
            right = x1
            break
        if np.isclose(k2, 0.0):
            right = x2
            break
        break

    width = max(0.0, float(right - left))
    return float(left), float(right), width


def window_metrics(x: np.ndarray, y: np.ndarray, xref: float, half_window_mm: float):
    mask = np.abs(x - xref) <= half_window_mm + 1e-12
    if not np.any(mask):
        return np.nan, np.nan, np.nan, 0

    xx = x[mask]
    yy = y[mask]
    ymean = float(np.trapz(yy, xx) / (xx[-1] - xx[0])) if len(xx) >= 2 and xx[-1] > xx[0] else float(np.mean(yy))
    return float(np.min(yy)), ymean, float(np.max(yy)), int(np.sum(mask))


def minmax_benefit(s: pd.Series):
    a = s.astype(float)
    if np.isclose(a.max(), a.min()):
        return pd.Series(np.ones(len(a)), index=a.index)
    return (a - a.min()) / (a.max() - a.min())


def minmax_cost(s: pd.Series):
    a = s.astype(float)
    if np.isclose(a.max(), a.min()):
        return pd.Series(np.ones(len(a)), index=a.index)
    return (a.max() - a) / (a.max() - a.min())


def main():
    ap = argparse.ArgumentParser(description="Evalúa población a partir de fit_curve.csv del postproceso híbrido.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--postprocess-summary", required=True)
    ap.add_argument("--out", default="population_evaluation.csv")
    ap.add_argument("--window-half-mm", type=float, default=1.0)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    post = pd.read_csv(args.postprocess_summary)

    if "individual_id" not in manifest.columns:
        if "name" in manifest.columns:
            manifest["individual_id"] = manifest["name"]
        else:
            raise RuntimeError("Manifest sin 'individual_id'")

    if "individual_id" not in post.columns:
        if "name" in post.columns:
            post["individual_id"] = post["name"]
        else:
            raise RuntimeError("Postprocess summary sin 'individual_id'")

    rows = []

    for _, rr in post.iterrows():
        ind = rr["individual_id"]

        if str(rr.get("post_status", "")) != "OK":
            rows.append({
                "individual_id": ind,
                "eval_status": "POST_FAIL"
            })
            continue

        fit_curve_csv = Path(rr["fit_curve_csv"])
        if not fit_curve_csv.exists():
            rows.append({
                "individual_id": ind,
                "eval_status": "FIT_CURVE_MISSING"
            })
            continue

        df = pd.read_csv(fit_curve_csv)

        x = df["x_mm"].to_numpy(float)
        f = df["F_fit_N"].to_numpy(float)
        k = df["K_shown_N_per_mm"].to_numpy(float)

        z_eq = choose_equilibrium_root(x, f)
        f_eq = interp_at(x, f, z_eq)
        k_eq = interp_at(x, k, z_eq)

        stab_left, stab_right, stab_width = positive_interval_around_reference(x, k, z_eq)
        kmin_w, kmean_w, kmax_w, nwin = window_metrics(x, k, z_eq, args.window_half_mm)

        rows.append({
            "individual_id": ind,
            "fit_curve_csv": str(fit_curve_csv),
            "z_eq_mm": z_eq,
            "F_eq_N": f_eq,
            "K_eq_N_per_mm": k_eq,
            "stable_left_mm": stab_left,
            "stable_right_mm": stab_right,
            "stable_width_mm": stab_width,
            "K_min_window_N_per_mm": kmin_w,
            "K_mean_window_N_per_mm": kmean_w,
            "K_max_window_N_per_mm": kmax_w,
            "window_half_mm": args.window_half_mm,
            "window_points": nwin,
            "eval_status": "OK"
        })

    eval_df = pd.DataFrame(rows)

    # unir geometría / parámetros del manifest
    keep_cols = [c for c in [
        "individual_id", "family", "rb1_mm", "rb2_mm", "hb_mm", "gap_pc_mm", "Ni", "I_eval_A"
    ] if c in manifest.columns]

    eval_df = manifest[keep_cols].merge(eval_df, on="individual_id", how="left")

    ok = eval_df["eval_status"].eq("OK")
    if ok.any():
        eval_df.loc[ok, "score_range"] = minmax_benefit(eval_df.loc[ok, "stable_width_mm"])
        eval_df.loc[ok, "score_stiffness"] = minmax_benefit(eval_df.loc[ok, "K_mean_window_N_per_mm"])
        eval_df.loc[ok, "score_current"] = minmax_cost(eval_df.loc[ok, "I_eval_A"].abs())
        eval_df.loc[ok, "score_centering"] = minmax_cost(eval_df.loc[ok, "z_eq_mm"].abs())

        eval_df.loc[ok, "score_total"] = (
            0.35 * eval_df.loc[ok, "score_range"] +
            0.40 * eval_df.loc[ok, "score_stiffness"] +
            0.20 * eval_df.loc[ok, "score_current"] +
            0.05 * eval_df.loc[ok, "score_centering"]
        )

    eval_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()