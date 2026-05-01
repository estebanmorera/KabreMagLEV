from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "10_poblacion"))

from design_space import (  # noqa: E402
    load_space_config,
    population_from_unit,
    variable_names,
)


SUPPORTED_ALGORITHMS = ("age2", "rvea", "ctaea", "sms", "nsga3", "random")


try:
    from pymoo.core.problem import Problem as _PymooProblem
except ImportError:  # pragma: no cover - random mode can work without pymoo
    _PymooProblem = object


class ExternalProblem(_PymooProblem):
    def __init__(self, n_var: int, n_obj: int, n_ieq_constr: int) -> None:
        if _PymooProblem is object:
            raise RuntimeError(
                "pymoo no esta instalado. Instala requirements_optional.txt o usa --algorithm random."
            )
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

    def _evaluate(self, x, out, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("Este problema se evalua externamente con Elmer/Gmsh.")


def _make_ref_dirs(n_obj: int, pop_size: int, seed: int):
    from pymoo.util.ref_dirs import get_reference_directions

    try:
        return get_reference_directions("energy", n_obj, n_points=pop_size, seed=seed)
    except Exception:
        partitions = 12 if n_obj <= 3 else 5
        return get_reference_directions("das-dennis", n_obj, n_partitions=partitions)


def _make_pymoo_algorithm(name: str, pop_size: int, n_obj: int, seed: int):
    name = name.lower()
    if name == "age2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2

        return AGEMOEA2(pop_size=pop_size)
    if name == "rvea":
        from pymoo.algorithms.moo.rvea import RVEA

        return RVEA(ref_dirs=_make_ref_dirs(n_obj, pop_size, seed))
    if name == "ctaea":
        from pymoo.algorithms.moo.ctaea import CTAEA

        return CTAEA(ref_dirs=_make_ref_dirs(n_obj, pop_size, seed))
    if name == "sms":
        from pymoo.algorithms.moo.sms import SMSEMOA

        return SMSEMOA(pop_size=pop_size)
    if name == "nsga3":
        from pymoo.algorithms.moo.nsga3 import NSGA3

        return NSGA3(ref_dirs=_make_ref_dirs(n_obj, pop_size, seed))
    raise ValueError(f"Algoritmo no soportado por pymoo: {name}")


class ExternalAskTellOptimizer:
    def __init__(
        self,
        algorithm_name: str,
        objective_names: list[str],
        constraint_names: list[str],
        pop_size: int,
        seed: int = 11,
        max_generations: int = 250,
        space_config: dict[str, Any] | None = None,
    ) -> None:
        algorithm_name = algorithm_name.lower()
        if algorithm_name not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algoritmo no soportado: {algorithm_name}")

        self.algorithm_name = algorithm_name
        self.objective_names = list(objective_names)
        self.constraint_names = list(constraint_names)
        self.pop_size = int(pop_size)
        self.seed = int(seed)
        self.max_generations = int(max_generations)
        self.space_config = space_config or load_space_config()
        self.generation = 0
        self.pending: dict[str, Any] | None = None
        self.archive: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(seed)
        self._algorithm = None
        self._problem = None

        if self.algorithm_name != "random":
            self._problem = ExternalProblem(
                n_var=len(variable_names(self.space_config)),
                n_obj=len(self.objective_names),
                n_ieq_constr=len(self.constraint_names),
            )
            self._algorithm = _make_pymoo_algorithm(
                self.algorithm_name,
                self.pop_size,
                len(self.objective_names),
                self.seed,
            )
            self._algorithm.setup(
                self._problem,
                termination=("n_gen", self.max_generations),
                seed=self.seed,
                verbose=False,
            )

    def ask_dataframe(self) -> pd.DataFrame:
        if self.pending is not None:
            raise RuntimeError("Hay una poblacion pendiente. Ejecuta tell antes de pedir otra.")

        if self.algorithm_name == "random":
            x = self._rng.random((self.pop_size, len(variable_names(self.space_config))))
            infills = None
        else:
            infills = self._algorithm.ask()
            x = np.asarray(infills.get("X"), dtype=float)

        df = population_from_unit(
            x,
            self.space_config,
            generation=self.generation,
            algorithm=self.algorithm_name,
            id_prefix=self.algorithm_name.upper(),
        )
        self.pending = {
            "x": x,
            "individual_ids": df["individual_id"].astype(str).tolist(),
            "infills": infills,
        }
        return df

    def tell_dataframe(self, evaluation: pd.DataFrame) -> None:
        if self.pending is None:
            raise RuntimeError("No hay poblacion pendiente para tell.")

        evaluation = evaluation.copy()
        evaluation["individual_id"] = evaluation["individual_id"].astype(str)
        evaluation = evaluation.set_index("individual_id", drop=False)

        ids = self.pending["individual_ids"]
        missing = [individual_id for individual_id in ids if individual_id not in evaluation.index]
        if missing:
            raise ValueError(f"Faltan resultados para individuos: {missing[:5]}")

        rows = evaluation.loc[ids]
        missing_cols = [
            col
            for col in self.objective_names + self.constraint_names
            if col not in rows.columns
        ]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en evaluation: {missing_cols}")

        f = rows[self.objective_names].astype(float).to_numpy()
        if not np.isfinite(f).all():
            raise ValueError("Hay objetivos NaN/Inf. Revisa optimizer_evaluation.csv")

        if self.constraint_names:
            g = rows[self.constraint_names].astype(float).to_numpy()
            if not np.isfinite(g).all():
                raise ValueError("Hay constraints NaN/Inf. Revisa optimizer_evaluation.csv")
        else:
            g = None

        if self.algorithm_name != "random":
            infills = self.pending["infills"]
            infills.set("F", f)
            if g is not None:
                infills.set("G", g)
            self._algorithm.tell(infills=infills)

        archive_rows = rows.reset_index(drop=True).copy()
        archive_rows["optimizer_generation"] = self.generation
        self.archive.extend(archive_rows.to_dict(orient="records"))
        self.pending = None
        self.generation += 1

    def archive_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.archive)

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm_name": self.algorithm_name,
            "generation": self.generation,
            "max_generations": self.max_generations,
            "pop_size": self.pop_size,
            "objective_names": self.objective_names,
            "constraint_names": self.constraint_names,
            "archive_rows": len(self.archive),
            "has_pending": self.pending is not None,
        }


def save_optimizer(optimizer: ExternalAskTellOptimizer, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(optimizer, f)
    return out


def load_optimizer(path: str | Path) -> ExternalAskTellOptimizer:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def create_optimizer(
    algorithm_name: str,
    objective_names: list[str],
    constraint_names: list[str],
    pop_size: int,
    seed: int = 11,
    max_generations: int = 250,
    space_json: str | Path | None = None,
) -> ExternalAskTellOptimizer:
    return ExternalAskTellOptimizer(
        algorithm_name=algorithm_name,
        objective_names=objective_names,
        constraint_names=constraint_names,
        pop_size=pop_size,
        seed=seed,
        max_generations=max_generations,
        space_config=load_space_config(space_json),
    )
