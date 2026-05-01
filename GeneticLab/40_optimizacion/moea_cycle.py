from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from algorithms import create_optimizer, load_optimizer, save_optimizer


DEFAULT_OBJECTIVES = [
    "objective_primary",
    "objective_abs_force0",
    "objective_nonlin",
    "objective_volume_m3",
]
DEFAULT_CONSTRAINTS = ["constraint_eval_failed", "constraint_geometry"]


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ciclo ask/tell para MOEA externo.")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Genera la siguiente poblacion candidata.")
    ask.add_argument("--state", required=True)
    ask.add_argument("--out", required=True)
    ask.add_argument("--algorithm", default="age2")
    ask.add_argument("--pop-size", type=int, default=16)
    ask.add_argument("--seed", type=int, default=11)
    ask.add_argument(
        "--max-generations",
        type=int,
        default=250,
        help="Horizonte n_gen para algoritmos que lo requieren, como RVEA.",
    )
    ask.add_argument("--space", default=None)
    ask.add_argument("--objectives", default=",".join(DEFAULT_OBJECTIVES))
    ask.add_argument("--constraints", default=",".join(DEFAULT_CONSTRAINTS))
    ask.add_argument("--overwrite-pending", action="store_true")

    tell = sub.add_parser("tell", help="Entrega resultados evaluados al optimizador.")
    tell.add_argument("--state", required=True)
    tell.add_argument("--evaluation", required=True)
    tell.add_argument("--archive-out", default="")

    status = sub.add_parser("status", help="Muestra resumen del optimizador.")
    status.add_argument("--state", required=True)

    return parser.parse_args()


def command_ask(args: argparse.Namespace) -> None:
    state = Path(args.state)
    if state.exists():
        optimizer = load_optimizer(state)
        if optimizer.pending is not None and args.overwrite_pending:
            optimizer.pending = None
    else:
        optimizer = create_optimizer(
            algorithm_name=args.algorithm,
            objective_names=_split_csv(args.objectives),
            constraint_names=_split_csv(args.constraints),
            pop_size=args.pop_size,
            seed=args.seed,
            max_generations=args.max_generations,
            space_json=args.space,
        )

    df = optimizer.ask_dataframe()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    save_optimizer(optimizer, state)
    print(f"Wrote ask population: {out}")
    print(json.dumps(optimizer.summary(), indent=2))


def command_tell(args: argparse.Namespace) -> None:
    optimizer = load_optimizer(args.state)
    evaluation = pd.read_csv(args.evaluation)
    optimizer.tell_dataframe(evaluation)
    save_optimizer(optimizer, args.state)
    if args.archive_out:
        archive = optimizer.archive_dataframe()
        out = Path(args.archive_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        archive.to_csv(out, index=False)
        print(f"Wrote archive: {out}")
    print(json.dumps(optimizer.summary(), indent=2))


def command_status(args: argparse.Namespace) -> None:
    optimizer = load_optimizer(args.state)
    print(json.dumps(optimizer.summary(), indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "ask":
        command_ask(args)
    elif args.command == "tell":
        command_tell(args)
    elif args.command == "status":
        command_status(args)


if __name__ == "__main__":
    main()
