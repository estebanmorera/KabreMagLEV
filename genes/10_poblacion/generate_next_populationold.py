#!/usr/bin/env python3
"""Generate a new population CSV from evaluated designs.

Typical use:

    python3 generate_next_population.py \
      --results mini_cycle/selected_designs.csv \
      --manifest mini_cycle/cases/population_manifest.csv \
      --out ga_cycle/gen_01/population.csv \
      --population-size 10 \
      --elite-count 2 \
      --mutation-rate 0.25 \
      --mutation-scale 0.15 \
      --id-prefix G01_I

The script is intentionally simple:
1. Reads the ranked results (`selected_designs.csv` or `population_evaluation.csv`).
2. Joins them with `population_manifest.csv` to recover the design genes.
3. Copies the best `elite_count` individuals unchanged.
4. Creates the remaining individuals using crossover + mutation.
5. Writes a new population CSV compatible with `write_population_cases.py`.
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
    "score_total",
    "stable_width_mm",
    "K_mean_window_N_per_mm",
    "K_min_window_N_per_mm",
    "z_eq_mm",
    "rank",
    "generation",
    "parent_a",
    "parent_b",
    "parent_kind",
    "elite",
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
    parser.add_argument(
        "--results",
        required=True,
        help="CSV con ranking final: selected_designs.csv o population_evaluation.csv",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="population_manifest.csv de la generación anterior",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="CSV de salida para la nueva población",
    )
    parser.add_argument(
        "--lineage-out",
        default="",
        help="CSV opcional con trazabilidad de padres; por defecto se crea junto al output",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=0,
        help="Tamaño de la nueva población. Si se omite, reutiliza el tamaño del manifest.",
    )
    parser.add_argument(
        "--elite-count",
        type=int,
        default=2,
        help="Número de mejores individuos que pasan sin cambios.",
    )
    parser.add_argument(
        "--parent-pool-size",
        type=int,
        default=0,
        help="Cuántos individuos usar como pool de padres; por defecto usa todos los resultados.",
    )
    parser.add_argument(
        "--gene-columns",
        default="",
        help="Lista separada por comas. Si se omite, usa las columnas del manifest excepto las auxiliares.",
    )
    parser.add_argument(
        "--keep-columns",
        default="",
        help="Columnas extra del manifest que quieras arrastrar aunque no sean genes.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.25,
        help="Probabilidad de mutación por gen numérico.",
    )
    parser.add_argument(
        "--mutation-scale",
        type=float,
        default=0.15,
        help="Tamaño relativo de la mutación respecto al rango observado del gen.",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.20,
        help="Expansión del crossover tipo BLX-alpha.",
    )
    parser.add_argument(
        "--categorical-mutation-rate",
        type=float,
        default=0.10,
        help="Probabilidad de cambiar un gen categórico a otro valor válido.",
    )
    parser.add_argument(
        "--id-prefix",
        default="I",
        help="Prefijo para los nuevos individual_id. Ejemplo: G01_I",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_ranked_population(results_path: Path, manifest_path: Path) -> pd.DataFrame:
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

    missing = merged[merged.isna().all(axis=1)]
    if not missing.empty:
        raise ValueError("La unión entre results y manifest produjo filas vacías completas.")

    if "score_total" in merged.columns:
        merged = merged.sort_values("score_total", ascending=False, kind="stable").reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    return merged


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

    cols = []
    for col in manifest_df.columns:
        if col in NON_GENE_COLUMNS:
            continue
        if col == "individual_id":
            continue
        cols.append(col)

    for col in keep_columns:
        if col not in cols and col in ranked_df.columns:
            cols.append(col)

    if not cols:
        raise ValueError(
            "No pude inferir genes desde el manifest. Usa --gene-columns para indicarlos manualmente."
        )

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
                NumericGene(
                    name=col,
                    low=low,
                    high=high,
                    integer_like=is_integer_like(series),
                )
            )
            continue

        values = [value for value in series.dropna().unique().tolist()]
        if not values:
            raise ValueError(f"La columna '{col}' no tiene valores válidos.")
        categorical_genes.append(CategoricalGene(name=col, values=values))

    return numeric_genes, categorical_genes


def build_parent_weights(n: int) -> np.ndarray:
    ranks = np.arange(n, 0, -1, dtype=float)
    weights = ranks / ranks.sum()
    return weights


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

    if math.isclose(low, high):
        child = low
    else:
        child = float(rng.uniform(low, high))

    if rng.random() < mutation_rate:
        sigma = max((gene.high - gene.low) * mutation_scale, 1e-9)
        child += float(rng.normal(loc=0.0, scale=sigma))

    child = float(np.clip(child, gene.low, gene.high))

    if gene.integer_like:
        return int(round(child))
    return child


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


def generate_next_population(
    ranked_df: pd.DataFrame,
    gene_columns: list[str],
    population_size: int,
    elite_count: int,
    parent_pool_size: int,
    mutation_rate: float,
    mutation_scale: float,
    blend_alpha: float,
    categorical_mutation_rate: float,
    id_prefix: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if population_size <= 0:
        raise ValueError("population_size debe ser > 0.")
    if elite_count < 0:
        raise ValueError("elite_count no puede ser negativo.")
    if elite_count > population_size:
        raise ValueError("elite_count no puede ser mayor que population_size.")
    if ranked_df.empty:
        raise ValueError("No hay individuos para crear la siguiente generación.")

    parent_pool = ranked_df.copy()
    if parent_pool_size > 0:
        parent_pool = parent_pool.head(parent_pool_size).copy()

    numeric_genes, categorical_genes = build_gene_models(parent_pool, gene_columns)
    weights = build_parent_weights(len(parent_pool))

    population_rows: list[dict[str, object]] = []
    lineage_rows: list[dict[str, object]] = []

    elites_df = ranked_df.head(elite_count)
    next_index = 1

    for _, elite in elites_df.iterrows():
        child = {"individual_id": make_individual_id(id_prefix, next_index)}
        for col in gene_columns:
            child[col] = elite[col]
        population_rows.append(child)
        lineage_rows.append(
            {
                "individual_id": child["individual_id"],
                "parent_a": elite["individual_id"],
                "parent_b": elite["individual_id"],
                "parent_kind": "elite",
                "elite": True,
            }
        )
        next_index += 1

    while len(population_rows) < population_size:
        parent_a, parent_b = choose_parents(parent_pool, weights, rng)
        child = {"individual_id": make_individual_id(id_prefix, next_index)}

        for gene in numeric_genes:
            a_value = float(parent_a[gene.name])
            b_value = float(parent_b[gene.name])
            child[gene.name] = crossover_numeric(
                gene=gene,
                a_value=a_value,
                b_value=b_value,
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

        population_rows.append(child)
        lineage_rows.append(
            {
                "individual_id": child["individual_id"],
                "parent_a": parent_a["individual_id"],
                "parent_b": parent_b["individual_id"],
                "parent_kind": "crossover",
                "elite": False,
            }
        )
        next_index += 1

    next_population_df = pd.DataFrame(population_rows)
    next_population_df = next_population_df[["individual_id", *gene_columns]]
    lineage_df = pd.DataFrame(lineage_rows)
    return next_population_df, lineage_df


def main() -> None:
    args = parse_args()

    results_path = Path(args.results).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    lineage_out = Path(args.lineage_out).expanduser().resolve() if args.lineage_out else out_path.with_name(
        out_path.stem + "_lineage.csv"
    )

    ranked_df = load_ranked_population(results_path, manifest_path)
    manifest_df = pd.read_csv(manifest_path)

    gene_columns = infer_gene_columns(
        ranked_df=ranked_df,
        manifest_df=manifest_df,
        explicit_gene_columns=parse_csv_list(args.gene_columns),
        keep_columns=parse_csv_list(args.keep_columns),
    )

    population_size = args.population_size or len(manifest_df)
    rng = np.random.default_rng(args.seed)

    next_population_df, lineage_df = generate_next_population(
        ranked_df=ranked_df,
        gene_columns=gene_columns,
        population_size=population_size,
        elite_count=args.elite_count,
        parent_pool_size=args.parent_pool_size,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        blend_alpha=args.blend_alpha,
        categorical_mutation_rate=args.categorical_mutation_rate,
        id_prefix=args.id_prefix,
        rng=rng,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_out.parent.mkdir(parents=True, exist_ok=True)

    next_population_df.to_csv(out_path, index=False)
    lineage_df.to_csv(lineage_out, index=False)

    print(f"results_csv      = {results_path}")
    print(f"manifest_csv     = {manifest_path}")
    print(f"population_out   = {out_path}")
    print(f"lineage_out      = {lineage_out}")
    print(f"population_size  = {len(next_population_df)}")
    print(f"elite_count      = {args.elite_count}")
    print(f"gene_columns     = {', '.join(gene_columns)}")
    print()
    print(next_population_df.head(min(10, len(next_population_df))).to_string(index=False))


if __name__ == "__main__":
    main()
