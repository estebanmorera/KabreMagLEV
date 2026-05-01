
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_METRIC_COLUMNS = {
    "stable_width_mm",
    "K_mean_window_N_per_mm",
    "K_min_window_N_per_mm",
    "z_eq_mm",
    "score_total",
    "score_range",
    "score_stiffness",
    "score_current",
    "rank",
}

DEFAULT_ID_COLUMNS = {
    "individual_id",
    "family",
    "generation",
    "parent_a",
    "parent_b",
    "origin",
}


def parse_bounds(bounds_text: str) -> Dict[str, List[float]]:
    """
    Formato:
      rb1_mm:11.0:15.5,rb2_mm:16.0:23.5,hb_mm:8.0:18.0,gap_pc_mm:4.0:5.5
    """
    out = {}
    if not bounds_text:
        return out
    for item in bounds_text.split(","):
        item = item.strip()
        if not item:
            continue
        name, lo, hi = item.split(":")
        out[name] = [float(lo), float(hi)]
    return out


def infer_design_vars(pop_df: pd.DataFrame, eval_df: pd.DataFrame, requested: Optional[List[str]]) -> List[str]:
    if requested:
        missing = [c for c in requested if c not in pop_df.columns]
        if missing:
            raise ValueError(f"Estas variables de diseño no están en population.csv: {missing}")
        return requested

    common = [c for c in pop_df.columns if c in eval_df.columns]
    candidates = []
    for c in common:
        if c in DEFAULT_ID_COLUMNS or c in DEFAULT_METRIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(pop_df[c]):
            candidates.append(c)

    # prioridad a nombres típicos
    preferred = [c for c in ["rb1_mm", "rb2_mm", "hb_mm", "gap_pc_mm", "Ni", "I_A", "I_eval_A"] if c in candidates]
    rest = [c for c in candidates if c not in preferred]
    out = preferred + rest

    if not out:
        raise ValueError(
            "No pude inferir variables de diseño. Usa --vars rb1_mm rb2_mm hb_mm gap_pc_mm ..."
        )
    return out


def tournament_select(df: pd.DataFrame, k: int, rng: random.Random) -> pd.Series:
    idx = rng.sample(list(df.index), k=min(k, len(df)))
    pool = df.loc[idx].copy()
    pool = pool.sort_values("score_total", ascending=False)
    return pool.iloc[0]


def clip_to_bounds(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mutate_value(value: float, lo: float, hi: float, sigma_frac: float, rng: random.Random) -> float:
    span = hi - lo
    sigma = sigma_frac * span
    new_val = value + rng.gauss(0.0, sigma)
    return clip_to_bounds(new_val, lo, hi)


def enforce_pair_constraints(child: Dict[str, float], bounds: Dict[str, List[float]], min_radial_thickness_mm: float):
    # Asegurar rb2 > rb1
    if "rb1_mm" in child and "rb2_mm" in child:
        rb1 = child["rb1_mm"]
        rb2 = child["rb2_mm"]
        if rb2 - rb1 < min_radial_thickness_mm:
            mid = 0.5 * (rb1 + rb2)
            rb1 = mid - 0.5 * min_radial_thickness_mm
            rb2 = mid + 0.5 * min_radial_thickness_mm

            if "rb1_mm" in bounds:
                rb1 = clip_to_bounds(rb1, bounds["rb1_mm"][0], bounds["rb1_mm"][1])
            if "rb2_mm" in bounds:
                rb2 = clip_to_bounds(rb2, bounds["rb2_mm"][0], bounds["rb2_mm"][1])

            # segunda pasada por si quedó pegado a límites
            if rb2 - rb1 < min_radial_thickness_mm:
                if "rb1_mm" in bounds and "rb2_mm" in bounds:
                    lo1, hi1 = bounds["rb1_mm"]
                    lo2, hi2 = bounds["rb2_mm"]
                    rb1 = clip_to_bounds(rb1, lo1, hi1)
                    rb2 = max(rb2, rb1 + min_radial_thickness_mm)
                    rb2 = clip_to_bounds(rb2, lo2, hi2)
                    rb1 = min(rb1, rb2 - min_radial_thickness_mm)
                    rb1 = clip_to_bounds(rb1, lo1, hi1)

            child["rb1_mm"] = rb1
            child["rb2_mm"] = rb2


def build_family_label(row: Dict[str, float]) -> str:
    rb1 = row.get("rb1_mm")
    rb2 = row.get("rb2_mm")
    hb = row.get("hb_mm")
    gap = row.get("gap_pc_mm")

    tags = []
    if rb1 is not None and rb2 is not None:
        thickness = rb2 - rb1
        mean_r = 0.5 * (rb1 + rb2)
        tags.append("compacta" if mean_r < 14.0 else "externa")
        tags.append("gruesa" if thickness >= 6.8 else "media")

    if hb is not None:
        if hb >= 16.0:
            tags.append("alta")
        elif hb <= 9.5:
            tags.append("baja")

    if gap is not None:
        if gap <= 4.4:
            tags.append("cercana")
        elif gap >= 5.0:
            tags.append("alejada")

    # quitar repetidos conservando orden
    out = []
    for t in tags:
        if t not in out:
            out.append(t)
    return "_".join(out) if out else "gen"


def main():
    ap = argparse.ArgumentParser(description="Genera una nueva población básica para el GA a partir de population.csv + evaluation.csv")
    ap.add_argument("--population", required=True, help="CSV de población base")
    ap.add_argument("--evaluation", required=True, help="CSV evaluado con score_total")
    ap.add_argument("--out", default="next_population.csv", help="CSV de salida")
    ap.add_argument("--vars", nargs="*", default=None, help="Variables de diseño. Ej: rb1_mm rb2_mm hb_mm gap_pc_mm")
    ap.add_argument("--bounds", default="", help="Bounds por variable: rb1_mm:11:15,rb2_mm:16:23,hb_mm:8:18,gap_pc_mm:4:5.5")
    ap.add_argument("--population-size", type=int, default=10)
    ap.add_argument("--elite-count", type=int, default=2)
    ap.add_argument("--tournament-size", type=int, default=3)
    ap.add_argument("--mutation-rate", type=float, default=0.35)
    ap.add_argument("--mutation-sigma-frac", type=float, default=0.10)
    ap.add_argument("--crossover-alpha", type=float, default=0.50, help="0.5 = promedio simple")
    ap.add_argument("--min-radial-thickness-mm", type=float, default=3.0)
    ap.add_argument("--generation", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    pop_df = pd.read_csv(args.population)
    eval_df = pd.read_csv(args.evaluation)

    if "individual_id" not in pop_df.columns or "individual_id" not in eval_df.columns:
        raise ValueError("Ambos CSV deben tener la columna individual_id")
    if "score_total" not in eval_df.columns:
        raise ValueError("evaluation.csv debe tener la columna score_total")

    merged = pop_df.merge(eval_df, on="individual_id", how="inner", suffixes=("", "_eval"))
    if merged.empty:
        raise RuntimeError("No hubo coincidencias entre population.csv y evaluation.csv por individual_id")

    design_vars = infer_design_vars(pop_df, eval_df, args.vars)

    # bounds
    cli_bounds = parse_bounds(args.bounds)
    bounds = {}
    for v in design_vars:
        if v in cli_bounds:
            bounds[v] = cli_bounds[v]
        else:
            col = pd.to_numeric(merged[v], errors="coerce")
            lo = float(np.nanmin(col))
            hi = float(np.nanmax(col))
            if math.isclose(lo, hi):
                delta = 0.05 * abs(lo) if abs(lo) > 1e-12 else 1.0
                lo -= delta
                hi += delta
            else:
                pad = 0.10 * (hi - lo)
                lo -= pad
                hi += pad
            # piso útil para gap
            if v == "gap_pc_mm":
                lo = max(lo, 4.0)
            bounds[v] = [lo, hi]

    ranked = merged.sort_values("score_total", ascending=False).reset_index(drop=True)

    out_rows = []

    # 1) Elites
    elite_count = min(args.elite_count, len(ranked), args.population_size)
    for i in range(elite_count):
        row = ranked.iloc[i]
        child = {v: float(row[v]) for v in design_vars}
        enforce_pair_constraints(child, bounds, args.min_radial_thickness_mm)
        out_rows.append({
            "individual_id": f"G{args.generation:02d}_E{i+1:02d}",
            **child,
            "family": row["family"] if "family" in row else build_family_label(child),
            "generation": args.generation,
            "parent_a": row["individual_id"],
            "parent_b": row["individual_id"],
            "origin": "elite",
        })

    # 2) Hijos por selección + cruza + mutación
    child_idx = 1
    while len(out_rows) < args.population_size:
        p1 = tournament_select(ranked, args.tournament_size, rng)
        p2 = tournament_select(ranked, args.tournament_size, rng)

        child = {}
        for v in design_vars:
            a = float(p1[v])
            b = float(p2[v])

            # cruza lineal sencilla
            alpha = args.crossover_alpha
            base = alpha * a + (1.0 - alpha) * b

            # un poco de mezcla extra para no caer siempre en el promedio exacto
            if rng.random() < 0.5:
                beta = rng.uniform(-0.15, 0.15)
                base = base + beta * (a - b)

            lo, hi = bounds[v]
            base = clip_to_bounds(base, lo, hi)

            if rng.random() < args.mutation_rate:
                base = mutate_value(base, lo, hi, args.mutation_sigma_frac, rng)

            child[v] = float(base)

        enforce_pair_constraints(child, bounds, args.min_radial_thickness_mm)

        out_rows.append({
            "individual_id": f"G{args.generation:02d}_C{child_idx:02d}",
            **child,
            "family": build_family_label(child),
            "generation": args.generation,
            "parent_a": p1["individual_id"],
            "parent_b": p2["individual_id"],
            "origin": "crossover_mutation",
        })
        child_idx += 1

    out_df = pd.DataFrame(out_rows)

    # deduplicado simple en variables de diseño
    if design_vars:
        out_df = out_df.drop_duplicates(subset=design_vars, keep="first").reset_index(drop=True)

    # rellenar si el dedup tumbó filas
    refill_guard = 0
    while len(out_df) < args.population_size and refill_guard < 200:
        refill_guard += 1
        p1 = tournament_select(ranked, args.tournament_size, rng)
        child = {}
        for v in design_vars:
            lo, hi = bounds[v]
            base = mutate_value(float(p1[v]), lo, hi, args.mutation_sigma_frac * 1.4, rng)
            child[v] = float(base)
        enforce_pair_constraints(child, bounds, args.min_radial_thickness_mm)

        row = {
            "individual_id": f"G{args.generation:02d}_R{refill_guard:02d}",
            **child,
            "family": build_family_label(child),
            "generation": args.generation,
            "parent_a": p1["individual_id"],
            "parent_b": "",
            "origin": "refill_mutation",
        }

        tmp = pd.concat([out_df, pd.DataFrame([row])], ignore_index=True)
        tmp = tmp.drop_duplicates(subset=design_vars, keep="first").reset_index(drop=True)
        out_df = tmp

    out_df = out_df.head(args.population_size).copy()
    out_df.to_csv(args.out, index=False)

    print("=== NEXT POPULATION ===")
    print(f"Población base: {args.population}")
    print(f"Evaluación:     {args.evaluation}")
    print(f"Salida:         {args.out}")
    print(f"Variables:      {design_vars}")
    print(f"Tamaño salida:  {len(out_df)}")
    print("\nBounds usados:")
    for k, (lo, hi) in bounds.items():
        print(f"  - {k}: [{lo:.6g}, {hi:.6g}]")

    print("\nTop de la nueva población:")
    cols = ["individual_id"] + design_vars + [c for c in ["family", "origin", "parent_a", "parent_b"] if c in out_df.columns]
    print(out_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
