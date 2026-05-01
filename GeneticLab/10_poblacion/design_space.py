from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "00_config" / "design_space_default.json"
)


def load_space_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def variable_names(config: dict[str, Any]) -> list[str]:
    return [row["name"] for row in config["variables"]]


def bounds_array(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([float(row["lower"]) for row in config["variables"]], dtype=float)
    upper = np.array([float(row["upper"]) for row in config["variables"]], dtype=float)
    return lower, upper


def lhs_unit(n_rows: int, n_dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty((n_rows, n_dim), dtype=float)
    for j in range(n_dim):
        perm = rng.permutation(n_rows)
        x[:, j] = (perm + rng.random(n_rows)) / n_rows
    return x


def random_unit(n_rows: int, n_dim: int, seed: int | None = None) -> np.ndarray:
    return np.random.default_rng(seed).random((n_rows, n_dim))


def unit_to_variable_values(x: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    lower, upper = bounds_array(config)
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, 1.0)
    values = lower + x * (upper - lower)
    return {name: float(value) for name, value in zip(variable_names(config), values)}


def _annulus_volume(ri: float, ro: float, h: float) -> float:
    return math.pi * max(ro * ro - ri * ri, 0.0) * max(h, 0.0)


def derive_design_columns(values: dict[str, float], config: dict[str, Any]) -> dict[str, Any]:
    mm = 1.0e-3
    fixed = config.get("fixed", {})
    limits = config.get("limits", {})

    r1 = values["r1_mm"] * mm
    r2 = r1 + values["t_inner_mm"] * mm
    r3 = r2 + values["gap_radial_mm"] * mm
    r4 = r3 + values["t_outer_mm"] * mm

    h0 = values["h0_mm"] * mm
    h1 = values["h1_mm"] * mm
    gap_z = values["gap_z_mm"] * mm
    gap_pc = values["gap_pc_mm"] * mm

    rb1 = r3
    rb2 = rb1 + values["coil_t_mm"] * mm
    hb = values["hb_mm"] * mm

    total_height = 2.0 * h0 + gap_z + gap_pc + hb
    volume_inner_m3 = 2.0 * _annulus_volume(r1, r2, h0)
    volume_outer_m3 = 2.0 * _annulus_volume(r3, r4, h1)
    volume_coil_m3 = _annulus_volume(rb1, rb2, hb)
    volume_total_m3 = volume_inner_m3 + volume_outer_m3 + volume_coil_m3

    g_values = []
    if "max_r4_mm" in limits:
        g_values.append((r4 / mm - limits["max_r4_mm"]) / limits["max_r4_mm"])
    if "max_rb2_mm" in limits:
        g_values.append((rb2 / mm - limits["max_rb2_mm"]) / limits["max_rb2_mm"])
    if "max_total_height_mm" in limits:
        g_values.append(
            (total_height / mm - limits["max_total_height_mm"])
            / limits["max_total_height_mm"]
        )
    if "min_gap_radial_mm" in limits:
        g_values.append(
            (limits["min_gap_radial_mm"] - values["gap_radial_mm"])
            / limits["min_gap_radial_mm"]
        )
    if "min_gap_z_mm" in limits:
        g_values.append((limits["min_gap_z_mm"] - values["gap_z_mm"]) / limits["min_gap_z_mm"])
    if "min_gap_pc_mm" in limits:
        g_values.append((limits["min_gap_pc_mm"] - values["gap_pc_mm"]) / limits["min_gap_pc_mm"])

    strict_geometry_ok = (
        0.0 < r1 < r2 < r3 < r4
        and 0.0 < rb1 < rb2
        and h0 > 0.0
        and h1 > 0.0
        and hb > 0.0
    )
    g_values.append(-1.0 if strict_geometry_ok else 1.0)
    constraint_geometry = max(g_values) if g_values else -1.0

    row = {
        **values,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "h0": h0,
        "h1": h1,
        "gap_z": gap_z,
        "gap_pc": gap_pc,
        "rb1": rb1,
        "rb2": rb2,
        "hb": hb,
        "Ni": int(fixed.get("Ni", 700)),
        "I_eval_A": float(fixed.get("I_eval_A", 3.0)),
        "total_height_m": total_height,
        "volume_inner_m3": volume_inner_m3,
        "volume_outer_m3": volume_outer_m3,
        "volume_coil_m3": volume_coil_m3,
        "volume_total_m3": volume_total_m3,
        "constraint_geometry": float(constraint_geometry),
        "geometry_status": "ok" if constraint_geometry <= 0.0 else "limit_violation",
    }
    return row


def unit_row_to_design(
    x: np.ndarray,
    config: dict[str, Any],
    individual_id: str,
    generation: int,
    algorithm: str,
) -> dict[str, Any]:
    values = unit_to_variable_values(x, config)
    row = derive_design_columns(values, config)
    row["individual_id"] = individual_id
    row["generation"] = int(generation)
    row["algorithm"] = algorithm
    for name, value in zip(variable_names(config), np.asarray(x, dtype=float)):
        row[f"x_{name}"] = float(value)
    return row


def population_from_unit(
    x: np.ndarray,
    config: dict[str, Any],
    generation: int = 0,
    algorithm: str = "manual",
    id_prefix: str | None = None,
) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x debe ser una matriz de tamano n_individuos x n_variables")
    prefix = id_prefix or algorithm.upper()
    rows = []
    for i, vector in enumerate(x):
        individual_id = f"{prefix}_G{generation:03d}_I{i:03d}"
        rows.append(unit_row_to_design(vector, config, individual_id, generation, algorithm))
    return pd.DataFrame(rows)


def make_initial_population(
    n_rows: int,
    config: dict[str, Any],
    seed: int | None = None,
    method: str = "lhs",
    generation: int = 0,
    algorithm: str = "initial",
) -> pd.DataFrame:
    n_dim = len(variable_names(config))
    if method == "lhs":
        x = lhs_unit(n_rows, n_dim, seed)
    elif method == "random":
        x = random_unit(n_rows, n_dim, seed)
    else:
        raise ValueError(f"Metodo de muestreo no soportado: {method}")
    return population_from_unit(x, config, generation=generation, algorithm=algorithm)


def write_population_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
