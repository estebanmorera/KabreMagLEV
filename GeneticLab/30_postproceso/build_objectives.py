from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "10_poblacion"))

from design_space import derive_design_columns, load_space_config  # noqa: E402


ALIASES = {
    "stiffness": [
        "stiffness_N_per_m",
        "k_N_per_m",
        "dF_dz_N_per_m",
        "force_slope_N_per_m",
        "k_eff_N_per_m",
        "stiffness",
    ],
    "score": ["score_total", "score_global", "fitness", "score"],
    "force0": [
        "force_at_0_N",
        "force_0_N",
        "F0_N",
        "F_eval_N",
        "force_center_N",
        "force_N",
    ],
    "nonlin": [
        "nonlinearity",
        "nonlinearity_N",
        "fit_rmse_N",
        "rmse_N",
        "polyfit_rmse_N",
        "force_rmse_N",
        "fit_rmse",
    ],
    "status": ["status", "run_status", "simulation_status", "eval_status"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte population_evaluation.csv en objetivos/constraints para el optimizador."
    )
    parser.add_argument("--population", required=True, help="CSV de poblacion evaluada.")
    parser.add_argument(
        "--evaluation",
        default="",
        help="population_evaluation.csv producido por genes/30_postproceso.",
    )
    parser.add_argument("--run-summary", default="", help="run_summary.csv opcional.")
    parser.add_argument("--postprocess-summary", default="", help="Resumen polyfit opcional.")
    parser.add_argument("--out", required=True, help="optimizer_evaluation.csv de salida.")
    parser.add_argument("--space", default=None, help="JSON de espacio de diseno.")
    parser.add_argument("--diagnostics-out", default="", help="JSON de diagnostico opcional.")
    parser.add_argument("--coil-resistance-ohm", type=float, default=9.0)
    return parser.parse_args()


def _read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _numeric_series(df: pd.DataFrame, col: str | None, default: float = np.nan) -> pd.Series:
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _status_failed(df: pd.DataFrame, status_col: str | None, primary: pd.Series) -> pd.Series:
    if status_col is None:
        return primary.isna()
    status = df[status_col].astype(str).str.lower()
    bad_words = ("fail", "error", "timeout", "missing", "not_completed", "nan")
    good_words = ("ok", "success", "complete", "done", "valid")
    failed = status.apply(lambda value: any(word in value for word in bad_words))
    explicitly_good = status.apply(lambda value: any(word in value for word in good_words))
    return failed | (~explicitly_good & primary.isna())


def _ensure_geometry_columns(population: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = population.copy()
    needed = {"volume_total_m3", "constraint_geometry"}
    if needed.issubset(df.columns):
        return df
    rows = []
    for _, row in df.iterrows():
        values = {name: float(row[name]) for name in [v["name"] for v in config["variables"]]}
        derived = derive_design_columns(values, config)
        rows.append(derived)
    derived_df = pd.DataFrame(rows)
    for col in derived_df.columns:
        if col not in df.columns:
            df[col] = derived_df[col]
    return df


def build_objectives(
    population_csv: str | Path,
    evaluation_csv: str | Path,
    out_csv: str | Path,
    space_json: str | Path | None = None,
    diagnostics_out: str | Path | None = None,
    coil_resistance_ohm: float = 9.0,
) -> pd.DataFrame:
    config = load_space_config(space_json)
    population = pd.read_csv(population_csv)
    population["individual_id"] = population["individual_id"].astype(str)
    population = _ensure_geometry_columns(population, config)

    evaluation = _read_csv_if_exists(evaluation_csv)
    if len(evaluation):
        evaluation["individual_id"] = evaluation["individual_id"].astype(str)
        df = population.merge(evaluation, on="individual_id", how="left", suffixes=("", "_eval"))
    else:
        df = population.copy()

    stiffness_col = _find_column(df, ALIASES["stiffness"])
    score_col = _find_column(df, ALIASES["score"])
    force0_col = _find_column(df, ALIASES["force0"])
    nonlin_col = _find_column(df, ALIASES["nonlin"])
    status_col = _find_column(df, ALIASES["status"])

    stiffness = _numeric_series(df, stiffness_col)
    score = _numeric_series(df, score_col)
    force0 = _numeric_series(df, force0_col, default=0.0).fillna(0.0)
    nonlin = _numeric_series(df, nonlin_col, default=0.0).fillna(0.0)

    if stiffness_col is not None:
        primary = -stiffness
        primary_source = stiffness_col
    elif score_col is not None:
        primary = -score
        primary_source = score_col
    else:
        primary = pd.Series(np.nan, index=df.index, dtype=float)
        primary_source = ""

    eval_failed = _status_failed(df, status_col, primary)
    geom = pd.to_numeric(df.get("constraint_geometry", -1.0), errors="coerce").fillna(1.0)
    penalty = 1.0e6 + 1.0e5 * geom.clip(lower=0.0)

    objective_primary = primary.where(~eval_failed & primary.notna(), penalty)
    objective_abs_force0 = force0.abs().where(~eval_failed, penalty)
    objective_nonlin = nonlin.abs().where(~eval_failed, penalty)
    objective_volume_m3 = pd.to_numeric(df["volume_total_m3"], errors="coerce").fillna(penalty)
    current = pd.to_numeric(df.get("I_eval_A", 3.0), errors="coerce").fillna(3.0)
    objective_power_W = current * current * float(coil_resistance_ohm)

    out = pd.DataFrame(
        {
            "individual_id": df["individual_id"].astype(str),
            "generation": df.get("generation", -1),
            "algorithm": df.get("algorithm", ""),
            "objective_primary": objective_primary,
            "objective_abs_force0": objective_abs_force0,
            "objective_nonlin": objective_nonlin,
            "objective_volume_m3": objective_volume_m3,
            "objective_power_W": objective_power_W,
            "constraint_eval_failed": np.where(eval_failed, 1.0, -1.0),
            "constraint_geometry": geom,
            "eval_failed": eval_failed,
            "geometry_status": df.get("geometry_status", ""),
            "primary_source": primary_source,
            "force0_source": force0_col or "",
            "nonlin_source": nonlin_col or "",
            "status_source": status_col or "",
        }
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    diagnostics = {
        "rows": int(len(out)),
        "primary_source": primary_source,
        "force0_source": force0_col or "",
        "nonlin_source": nonlin_col or "",
        "status_source": status_col or "",
        "failed_rows": int(out["eval_failed"].sum()),
        "notes": [
            "Todos los objetivos se minimizan.",
            "constraint_* debe ser <= 0 para ser factible.",
            "Si primary_source esta vacio, ajusta ALIASES o el evaluador existente.",
        ],
    }
    if diagnostics_out:
        diag_path = Path(diagnostics_out)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    return out


def main() -> None:
    args = parse_args()
    diagnostics_out = args.diagnostics_out or str(Path(args.out).with_suffix(".diagnostics.json"))
    build_objectives(
        population_csv=args.population,
        evaluation_csv=args.evaluation,
        out_csv=args.out,
        space_json=args.space,
        diagnostics_out=diagnostics_out,
        coil_resistance_ohm=args.coil_resistance_ohm,
    )


if __name__ == "__main__":
    main()
