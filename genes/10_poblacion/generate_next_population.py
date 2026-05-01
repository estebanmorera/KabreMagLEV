#!/usr/bin/env python3
"""Generate the next genetic population from one generation or a history pool.

Backwards-compatible single-generation use:

    python3 generate_next_population.py \
      --results Genetic1/results/generation2/04_selection/population_evaluation.csv \
      --manifest Genetic1/results/generation2/01_population/population_manifest.csv \
      --out Genetic1/results/generation3/01_population/population.csv

Recommended historical use:

    python3 generate_next_population.py \
      --history-evaluation Genetic1/results/generation1/04_selection/population_evaluation.csv \
      --history-manifest Genetic1/results/generation1/01_population/population_manifest.csv \
      --history-evaluation Genetic1/results/generation2/04_selection/population_evaluation.csv \
      --history-manifest Genetic1/results/generation2/01_population/population_manifest.csv \
      --history-out Genetic1/results/all_evaluated.csv \
      --out Genetic1/results/generation3/01_population/population.csv \
      --population-size 15 \
      --elite-count 2 \
      --parent-pool-size 8 \
      --local-search-count 5 \
      --id-prefix G03_I
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NON_GENE_COLUMNS = {
    "individual_id",
    "case_dir",
    "geo_path",
    "sif_path",
    "definition_path",
    "score_total",
    "score_global",
    "global_rank",
    "stable_width_mm",
    "K_mean_window_N_per_mm",
    "K_min_window_N_per_mm",
    "z_eq_mm",
    "rank",
    "generation",
    "source_generation",
    "source_results",
    "source_manifest",
    "parent_a",
    "parent_b",
    "parent_kind",
    "origin",
    "elite",
    "Ni",
    "I_eval_A",
}


@dataclass
class NumericGene:
    name: str
    low: float
    high: float
    integer_like: bool


@dataclass
class CategoricalGene:
    name: str
    values: list[object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="", help="CSV de evaluacion de una sola generacion.")
    parser.add_argument("--manifest", default="", help="Manifest de una sola generacion.")
    parser.add_argument(
        "--history-evaluation",
        action="append",
        default=[],
        help="CSV de evaluacion historica. Repetir una vez por generacion.",
    )
    parser.add_argument(
        "--history-manifest",
        action="append",
        default=[],
        help="Manifest historico. Repetir en el mismo orden que --history-evaluation.",
    )
    parser.add_argument("--history-out", default="", help="CSV opcional con el historico unido y score_global.")
    parser.add_argument("--out", required=True, help="CSV de salida para la nueva poblacion.")
    parser.add_argument(
        "--lineage-out",
        default="",
        help="CSV opcional con trazabilidad; por defecto se crea junto al output.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=0,
        help="Tamano de salida. Si se omite, reutiliza el tamano del primer manifest.",
    )
    parser.add_argument("--elite-count", type=int, default=2, help="Individuos top copiados sin cambio.")
    parser.add_argument(
        "--parent-pool-size",
        type=int,
        default=0,
        help="Cuantos top historicos usar como pool de padres. 0 usa todos.",
    )
    parser.add_argument(
        "--local-search-count",
        type=int,
        default=0,
        help="Numero de hijos por perturbacion local alrededor de los mejores historicos.",
    )
    parser.add_argument(
        "--local-search-scale",
        type=float,
        default=0.07,
        help="Sigma relativo del local search respecto al rango historico de cada gen.",
    )
    parser.add_argument(
        "--gene-columns",
        default="",
        help="Lista separada por comas. Si se omite, infiere genes desde el manifest.",
    )
    parser.add_argument(
        "--keep-columns",
        default="",
        help="Columnas extra que quieres arrastrar como genes aunque normalmente se excluirian.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.30, help="Probabilidad de mutacion numerica.")
    parser.add_argument("--mutation-scale", type=float, default=0.12, help="Sigma relativo de mutacion.")
    parser.add_argument("--blend-alpha", type=float, default=0.15, help="Expansion del crossover BLX-alpha.")
    parser.add_argument(
        "--categorical-mutation-rate",
        type=float,
        default=0.10,
        help="Probabilidad de mutar genes categoricos.",
    )
    parser.add_argument("--id-prefix", default="I", help="Prefijo de individual_id. Ejemplo: G03_I")
    parser.add_argument("--generation", type=int, default=0, help="Numero de generacion para metadatos.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument(
        "--score-mode",
        choices=["global", "existing"],
        default="global",
        help="global recalcula score comparable; existing usa score_total/score_global existente.",
    )
    parser.add_argument("--stable-weight", type=float, default=0.35)
    parser.add_argument("--k-mean-weight", type=float, default=0.25)
    parser.add_argument("--k-min-weight", type=float, default=0.25)
    parser.add_argument("--z-eq-weight", type=float, default=0.15)
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_history_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.history_evaluation or args.history_manifest:
        if len(args.history_evaluation) != len(args.history_manifest):
            raise ValueError("--history-evaluation y --history-manifest deben tener la misma cantidad.")
        return [
            (Path(eval_path).expanduser().resolve(), Path(manifest_path).expanduser().resolve())
            for eval_path, manifest_path in zip(args.history_evaluation, args.history_manifest)
        ]

    if not args.results or not args.manifest:
        raise ValueError("Usa --results/--manifest o pares --history-evaluation/--history-manifest.")

    return [(Path(args.results).expanduser().resolve(), Path(args.manifest).expanduser().resolve())]


def load_population_pair(results_path: Path, manifest_path: Path, source_generation: str) -> pd.DataFrame:
    results_df = pd.read_csv(results_path)
    manifest_df = pd.read_csv(manifest_path)

    if "individual_id" not in results_df.columns:
        raise ValueError(f"{results_path} no contiene la columna 'individual_id'.")
    if "individual_id" not in manifest_df.columns:
        raise ValueError(f"{manifest_path} no contiene la columna 'individual_id'.")

    extra_manifest_cols = [
        col for col in manifest_df.columns if col == "individual_id" or col not in results_df.columns
    ]
    merged = results_df.merge(
        manifest_df[extra_manifest_cols],
        on="individual_id",
        how="left",
        validate="one_to_one",
    )
    merged["source_generation"] = source_generation
    merged["source_results"] = str(results_path)
    merged["source_manifest"] = str(manifest_path)
    return merged


def normalize_higher_is_better(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or math.isclose(float(lo), float(hi)):
        return pd.Series(0.5, index=series.index)
    return (values - lo) / (hi - lo)


def add_global_score(
    df: pd.DataFrame,
    stable_weight: float,
    k_mean_weight: float,
    k_min_weight: float,
    z_eq_weight: float,
) -> pd.DataFrame:
    out = df.copy()
    weighted_parts: list[pd.Series] = []
    weights: list[float] = []

    metric_specs = [
        ("stable_width_mm", stable_weight, True),
        ("K_mean_window_N_per_mm", k_mean_weight, True),
        ("K_min_window_N_per_mm", k_min_weight, True),
        ("z_eq_mm", z_eq_weight, False),
    ]

    for col, weight, higher_is_better in metric_specs:
        if weight <= 0 or col not in out.columns:
            continue

        if higher_is_better:
            score_part = normalize_higher_is_better(out[col])
        else:
            score_part = 1.0 - normalize_higher_is_better(pd.to_numeric(out[col], errors="coerce").abs())

        weighted_parts.append(score_part.fillna(0.0) * weight)
        weights.append(weight)

    if not weighted_parts:
        if "score_total" not in out.columns:
            raise ValueError("No hay metricas para score_global ni columna score_total.")
        out["score_global"] = pd.to_numeric(out["score_total"], errors="coerce").fillna(0.0)
    else:
        out["score_global"] = sum(weighted_parts) / sum(weights)

    out = out.sort_values("score_global", ascending=False, kind="stable").reset_index(drop=True)
    out["global_rank"] = np.arange(1, len(out) + 1)
    return out


def sort_existing_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "score_global" in out.columns:
        score_col = "score_global"
    elif "score_total" in out.columns:
        score_col = "score_total"
    else:
        return out.reset_index(drop=True)
    return out.sort_values(score_col, ascending=False, kind="stable").reset_index(drop=True)


def infer_gene_columns(
    ranked_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    explicit_gene_columns: list[str],
    keep_columns: list[str],
) -> list[str]:
    if explicit_gene_columns:
        missing = [col for col in explicit_gene_columns if col not in ranked_df.columns]
        if missing:
            raise ValueError(f"Estas columnas pedidas no existen en los CSV: {missing}")
        return explicit_gene_columns

    keep_set = set(keep_columns)
    cols = []
    for col in manifest_df.columns:
        if col == "individual_id":
            continue
        if col in NON_GENE_COLUMNS and col not in keep_set:
            continue
        if col not in ranked_df.columns:
            continue
        cols.append(col)

    for col in keep_columns:
        if col not in cols and col in ranked_df.columns:
            cols.append(col)

    if not cols:
        raise ValueError("No pude inferir genes desde el manifest. Usa --gene-columns.")

    return cols


def is_integer_like(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        return False
    return np.allclose(values, np.round(values), atol=1e-9)


def build_gene_models(base_df: pd.DataFrame, gene_columns: Iterable[str]) -> tuple[list[NumericGene], list[CategoricalGene]]:
    numeric_genes: list[NumericGene] = []
    categorical_genes: list[CategoricalGene] = []

    for col in gene_columns:
        series = base_df[col]
        numeric = pd.to_numeric(series, errors="coerce")

        if numeric.notna().all():
            low = float(numeric.min())
            high = float(numeric.max())
            numeric_genes.append(
                NumericGene(name=col, low=low, high=high, integer_like=is_integer_like(series))
            )
        else:
            values = [value for value in series.dropna().unique().tolist()]
            if not values:
                raise ValueError(f"La columna '{col}' no tiene valores validos.")
            categorical_genes.append(CategoricalGene(name=col, values=values))

    return numeric_genes, categorical_genes


def build_parent_weights(n: int) -> np.ndarray:
    ranks = np.arange(n, 0, -1, dtype=float)
    return ranks / ranks.sum()


def make_individual_id(prefix: str, index_1based: int) -> str:
    return f"{prefix}{index_1based:03d}"


def choose_parents(
    parent_pool: pd.DataFrame,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    if len(parent_pool) == 1:
        row = parent_pool.iloc[0]
        return row, row
    idx = rng.choice(len(parent_pool), size=2, replace=False, p=weights)
    return parent_pool.iloc[int(idx[0])], parent_pool.iloc[int(idx[1])]


def crossover_numeric(
    gene: NumericGene,
    a_value: float,
    b_value: float,
    mutation_rate: float,
    mutation_scale: float,
    blend_alpha: float,
    rng: np.random.Generator,
) -> float | int:
    low_parent = min(a_value, b_value)
    high_parent = max(a_value, b_value)
    span = high_parent - low_parent
    low = low_parent - blend_alpha * span
    high = high_parent + blend_alpha * span

    child = low if math.isclose(low, high) else float(rng.uniform(low, high))

    if rng.random() < mutation_rate:
        sigma = max((gene.high - gene.low) * mutation_scale, 1e-9)
        child += float(rng.normal(loc=0.0, scale=sigma))

    child = float(np.clip(child, gene.low, gene.high))
    return int(round(child)) if gene.integer_like else child


def local_search_numeric(
    gene: NumericGene,
    parent_value: float,
    local_search_scale: float,
    rng: np.random.Generator,
) -> float | int:
    sigma = max((gene.high - gene.low) * local_search_scale, 1e-9)
    child = float(parent_value) + float(rng.normal(loc=0.0, scale=sigma))
    child = float(np.clip(child, gene.low, gene.high))
    return int(round(child)) if gene.integer_like else child


def crossover_categorical(
    gene: CategoricalGene,
    a_value: object,
    b_value: object,
    mutation_rate: float,
    rng: np.random.Generator,
) -> object:
    if rng.random() < mutation_rate:
        return rng.choice(gene.values)
    return a_value if rng.random() < 0.5 else b_value


def create_child_from_parent(
    parent: pd.Series,
    numeric_genes: list[NumericGene],
    categorical_genes: list[CategoricalGene],
    gene_columns: list[str],
    local_search_scale: float,
    categorical_mutation_rate: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    child: dict[str, object] = {}

    for gene in numeric_genes:
        child[gene.name] = local_search_numeric(gene, float(parent[gene.name]), local_search_scale, rng)

    for gene in categorical_genes:
        value = parent[gene.name]
        if rng.random() < categorical_mutation_rate:
            value = rng.choice(gene.values)
        child[gene.name] = value

    return {col: child[col] for col in gene_columns}


def create_child_from_crossover(
    parent_a: pd.Series,
    parent_b: pd.Series,
    numeric_genes: list[NumericGene],
    categorical_genes: list[CategoricalGene],
    gene_columns: list[str],
    mutation_rate: float,
    mutation_scale: float,
    blend_alpha: float,
    categorical_mutation_rate: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    child: dict[str, object] = {}

    for gene in numeric_genes:
        child[gene.name] = crossover_numeric(
            gene=gene,
            a_value=float(parent_a[gene.name]),
            b_value=float(parent_b[gene.name]),
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            blend_alpha=blend_alpha,
            rng=rng,
        )

    for gene in categorical_genes:
        child[gene.name] = crossover_categorical(
            gene=gene,
            a_value=parent_a[gene.name],
            b_value=parent_b[gene.name],
            mutation_rate=categorical_mutation_rate,
            rng=rng,
        )

    return {col: child[col] for col in gene_columns}


def add_row_metadata(
    child: dict[str, object],
    individual_id: str,
    generation: int,
    parent_a: str,
    parent_b: str,
    origin: str,
) -> dict[str, object]:
    out = {"individual_id": individual_id, **child}
    if generation:
        out["generation"] = generation
    out["parent_a"] = parent_a
    out["parent_b"] = parent_b
    out["origin"] = origin
    return out


def generate_next_population(
    ranked_df: pd.DataFrame,
    gene_columns: list[str],
    population_size: int,
    elite_count: int,
    parent_pool_size: int,
    local_search_count: int,
    local_search_scale: float,
    mutation_rate: float,
    mutation_scale: float,
    blend_alpha: float,
    categorical_mutation_rate: float,
    id_prefix: str,
    generation: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if population_size <= 0:
        raise ValueError("population_size debe ser > 0.")
    if elite_count < 0 or elite_count > population_size:
        raise ValueError("elite_count debe estar entre 0 y population_size.")
    if ranked_df.empty:
        raise ValueError("No hay individuos para crear la siguiente generacion.")

    parent_pool = ranked_df.copy()
    if parent_pool_size > 0:
        parent_pool = parent_pool.head(parent_pool_size).copy()

    numeric_genes, categorical_genes = build_gene_models(ranked_df, gene_columns)
    weights = build_parent_weights(len(parent_pool))

    population_rows: list[dict[str, object]] = []
    lineage_rows: list[dict[str, object]] = []
    next_index = 1

    for _, elite in ranked_df.head(elite_count).iterrows():
        child = {col: elite[col] for col in gene_columns}
        individual_id = make_individual_id(id_prefix, next_index)
        row = add_row_metadata(
            child=child,
            individual_id=individual_id,
            generation=generation,
            parent_a=str(elite["individual_id"]),
            parent_b=str(elite["individual_id"]),
            origin="elite",
        )
        population_rows.append(row)
        lineage_rows.append(
            {
                "individual_id": individual_id,
                "parent_a": elite["individual_id"],
                "parent_b": elite["individual_id"],
                "parent_kind": "elite",
                "elite": True,
            }
        )
        next_index += 1

    local_search_to_make = min(local_search_count, population_size - len(population_rows))
    for i in range(local_search_to_make):
        parent = parent_pool.iloc[i % len(parent_pool)]
        child = create_child_from_parent(
            parent=parent,
            numeric_genes=numeric_genes,
            categorical_genes=categorical_genes,
            gene_columns=gene_columns,
            local_search_scale=local_search_scale,
            categorical_mutation_rate=categorical_mutation_rate,
            rng=rng,
        )
        individual_id = make_individual_id(id_prefix, next_index)
        row = add_row_metadata(
            child=child,
            individual_id=individual_id,
            generation=generation,
            parent_a=str(parent["individual_id"]),
            parent_b=str(parent["individual_id"]),
            origin="local_search",
        )
        population_rows.append(row)
        lineage_rows.append(
            {
                "individual_id": individual_id,
                "parent_a": parent["individual_id"],
                "parent_b": parent["individual_id"],
                "parent_kind": "local_search",
                "elite": False,
            }
        )
        next_index += 1

    while len(population_rows) < population_size:
        parent_a, parent_b = choose_parents(parent_pool, weights, rng)
        child = create_child_from_crossover(
            parent_a=parent_a,
            parent_b=parent_b,
            numeric_genes=numeric_genes,
            categorical_genes=categorical_genes,
            gene_columns=gene_columns,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            blend_alpha=blend_alpha,
            categorical_mutation_rate=categorical_mutation_rate,
            rng=rng,
        )
        individual_id = make_individual_id(id_prefix, next_index)
        row = add_row_metadata(
            child=child,
            individual_id=individual_id,
            generation=generation,
            parent_a=str(parent_a["individual_id"]),
            parent_b=str(parent_b["individual_id"]),
            origin="crossover_mutation",
        )
        population_rows.append(row)
        lineage_rows.append(
            {
                "individual_id": individual_id,
                "parent_a": parent_a["individual_id"],
                "parent_b": parent_b["individual_id"],
                "parent_kind": "crossover_mutation",
                "elite": False,
            }
        )
        next_index += 1

    population_df = pd.DataFrame(population_rows)
    metadata_cols = [col for col in ["generation", "parent_a", "parent_b", "origin"] if col in population_df.columns]
    population_df = population_df[["individual_id", *gene_columns, *metadata_cols]]
    lineage_df = pd.DataFrame(lineage_rows)
    return population_df, lineage_df


def main() -> None:
    args = parse_args()

    pairs = build_history_pairs(args)
    history_parts = []
    manifest_parts = []
    for idx, (results_path, manifest_path) in enumerate(pairs, 1):
        source_generation = manifest_path.parent.parent.name if "generation" in str(manifest_path) else f"history{idx}"
        history_parts.append(load_population_pair(results_path, manifest_path, source_generation))
        manifest_parts.append(pd.read_csv(manifest_path))

    history_df = pd.concat(history_parts, ignore_index=True)
    manifest_df = pd.concat(manifest_parts, ignore_index=True)

    if args.score_mode == "global":
        ranked_df = add_global_score(
            history_df,
            stable_weight=args.stable_weight,
            k_mean_weight=args.k_mean_weight,
            k_min_weight=args.k_min_weight,
            z_eq_weight=args.z_eq_weight,
        )
    else:
        ranked_df = sort_existing_score(history_df)

    if args.history_out:
        history_out = Path(args.history_out).expanduser().resolve()
        history_out.parent.mkdir(parents=True, exist_ok=True)
        ranked_df.to_csv(history_out, index=False)

    gene_columns = infer_gene_columns(
        ranked_df=ranked_df,
        manifest_df=manifest_df,
        explicit_gene_columns=parse_csv_list(args.gene_columns),
        keep_columns=parse_csv_list(args.keep_columns),
    )

    population_size = args.population_size or len(manifest_parts[0])
    rng = np.random.default_rng(args.seed)

    next_population_df, lineage_df = generate_next_population(
        ranked_df=ranked_df,
        gene_columns=gene_columns,
        population_size=population_size,
        elite_count=args.elite_count,
        parent_pool_size=args.parent_pool_size,
        local_search_count=args.local_search_count,
        local_search_scale=args.local_search_scale,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        blend_alpha=args.blend_alpha,
        categorical_mutation_rate=args.categorical_mutation_rate,
        id_prefix=args.id_prefix,
        generation=args.generation,
        rng=rng,
    )

    out_path = Path(args.out).expanduser().resolve()
    lineage_out = Path(args.lineage_out).expanduser().resolve() if args.lineage_out else out_path.with_name(
        out_path.stem + "_lineage.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_out.parent.mkdir(parents=True, exist_ok=True)

    next_population_df.to_csv(out_path, index=False)
    lineage_df.to_csv(lineage_out, index=False)

    print("--- NEXT POPULATION ---")
    print(f"history_pairs     = {len(pairs)}")
    print(f"population_out    = {out_path}")
    print(f"lineage_out       = {lineage_out}")
    print(f"population_size   = {len(next_population_df)}")
    print(f"elite_count       = {args.elite_count}")
    print(f"parent_pool_size  = {args.parent_pool_size or len(ranked_df)}")
    print(f"local_search      = {args.local_search_count}")
    print(f"gene_columns      = {', '.join(gene_columns)}")
    print()
    print(next_population_df.head(min(15, len(next_population_df))).to_string(index=False))


if __name__ == "__main__":
    main()
