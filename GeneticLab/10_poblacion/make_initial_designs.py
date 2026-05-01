from __future__ import annotations

import argparse
from pathlib import Path

from design_space import load_space_config, make_initial_population, write_population_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera una poblacion inicial LHS/random compatible con genes."
    )
    parser.add_argument("--out", required=True, help="CSV de salida.")
    parser.add_argument("--n", type=int, default=16, help="Numero de individuos.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--method", choices=["lhs", "random"], default="lhs")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--algorithm", default="initial")
    parser.add_argument(
        "--space",
        default=None,
        help="JSON de espacio de diseno. Usa 00_config/design_space_default.json si se omite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_space_config(args.space)
    df = make_initial_population(
        n_rows=args.n,
        config=config,
        seed=args.seed,
        method=args.method,
        generation=args.generation,
        algorithm=args.algorithm,
    )
    out = write_population_csv(df, Path(args.out))
    print(f"Wrote {len(df)} designs -> {out}")
    print(df[["individual_id", "r1", "r2", "r3", "r4", "h0", "h1", "gap_z", "rb1", "rb2", "hb"]].head())


if __name__ == "__main__":
    main()
