from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_MODULE_PREAMBLE = """
module purge
module load elmerfem/9.0
module load gmsh/4.15.0
module load lapack/3.12.1
export LD_LIBRARY_PATH=/work/jmorera/compat_gfortran3:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HYDRA_LAUNCHER=fork
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta una poblacion GeneticLab usando el pipeline existente de genes."
    )
    parser.add_argument("--project", default="/work/jmorera/Genes")
    parser.add_argument("--lab-root", default="")
    parser.add_argument("--population", required=True)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--experiment-root", default="")
    parser.add_argument("--execute", action="store_true", help="Ejecuta comandos externos.")
    parser.add_argument("--axis", default="dz")
    parser.add_argument("--start-mm", type=float, default=-3.0)
    parser.add_argument("--end-mm", type=float, default=3.0)
    parser.add_argument("--n-points", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=4)
    parser.add_argument("--nprocs", type=int, default=20)
    parser.add_argument("--inner-launcher", default="mpirun")
    parser.add_argument("--bind", default="")
    parser.add_argument("--partition-method", default="metiskway")
    parser.add_argument("--gmsh-threads", type=int, default=1)
    parser.add_argument("--gmsh-launcher", default="serial")
    parser.add_argument("--gmsh-mpi-procs", type=int, default=1)
    parser.add_argument("--gmsh-extra-args", default="")
    parser.add_argument("--python-exe-remote", default="python3")
    parser.add_argument("--max-individuals", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--auto-rescue", action="store_true")
    parser.add_argument("--rescue-frac", type=float, default=0.5)
    parser.add_argument("--skip-first-node", action="store_true")
    parser.add_argument("--skip-completed", dest="skip_completed", action="store_true", default=True)
    parser.add_argument("--no-skip-completed", dest="skip_completed", action="store_false")
    return parser.parse_args()


def run_shell(cmd: str, cwd: Path, use_modules: bool, execute: bool) -> str:
    full_cmd = f"{DEFAULT_MODULE_PREAMBLE}\n{cmd}" if use_modules else cmd
    if not execute:
        return "DRY_RUN command:\n" + full_cmd
    proc = subprocess.run(
        ["bash", "-lc", full_cmd],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    parts = []
    if proc.stdout:
        parts.append("STDOUT:\n" + proc.stdout)
    if proc.stderr:
        parts.append("STDERR:\n" + proc.stderr)
    if proc.returncode != 0:
        parts.append(f"[returncode={proc.returncode}]")
    return "\n".join(parts)


def step_for_n_points(start_mm: float, end_mm: float, n_points: int) -> float:
    if n_points < 2:
        raise ValueError("--n-points debe ser >= 2")
    return round(abs(end_mm - start_mm) / (n_points - 1), 8)


def expected_rows_for_range(start_mm: float, end_mm: float, step_mm: float) -> int:
    return int(math.floor(abs(end_mm - start_mm) / step_mm)) + 1


def ensure_existing(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Faltan rutas requeridas:\n" + "\n".join(missing))


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    lab_root = Path(args.lab_root).resolve() if args.lab_root else project / "GeneticLab"
    run_label = args.run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = (
        Path(args.experiment_root).resolve()
        if args.experiment_root
        else lab_root / "experiments" / run_label
    )

    genes_root = project / "genes"
    templates_dir = genes_root / "00_templates_simulacion"
    population_scripts_dir = genes_root / "10_poblacion"
    execution_dir = genes_root / "20_ejecucion"
    postprocess_dir = genes_root / "30_postproceso"

    geo_template = templates_dir / "StepsHTX.geo"
    sif_template = templates_dir / "P1low.sif"
    def_template = templates_dir / "HMB_circuit.definition"
    write_cases_py = population_scripts_dir / "write_population_cases.py"
    runner_mpi = execution_dir / "sweep_mpi.py"
    eval_one_src = execution_dir / "genes_slurm_eval_individual.py"
    alloc_runner_src = execution_dir / "genes_slurm_run_current_allocation.py"
    collect_src = postprocess_dir / "genes_slurm_collect_run_summary.py"
    polyfit_py = postprocess_dir / "polyfit_energy_force_stiffness_hybrid.py"
    postprocess_polyfit_py = postprocess_dir / "postprocess_population_polyfit.py"
    evaluate_polyfit_py = postprocess_dir / "evaluate_population_polyfit.py"
    select_py = postprocess_dir / "select_top_designs.py"

    ensure_existing(
        [
            Path(args.population),
            geo_template,
            sif_template,
            def_template,
            write_cases_py,
            runner_mpi,
            eval_one_src,
            alloc_runner_src,
            collect_src,
            polyfit_py,
            postprocess_polyfit_py,
            evaluate_polyfit_py,
            select_py,
        ]
    )

    cases_dir = experiment_root / "cases"
    runs_dir = experiment_root / "runs"
    results_dir = experiment_root / "results"
    scripts_dir = experiment_root / "scripts"
    dispatch_logs_dir = experiment_root / "dispatch_logs"
    pop_results_dir = results_dir / "01_population"
    exec_results_dir = results_dir / "02_execution"
    post_results_dir = results_dir / "03_postprocess"
    sel_results_dir = results_dir / "04_selection"
    opt_results_dir = results_dir / "05_optimizer"

    for folder in [
        cases_dir,
        runs_dir,
        results_dir,
        scripts_dir,
        dispatch_logs_dir,
        pop_results_dir,
        exec_results_dir,
        post_results_dir,
        sel_results_dir,
        opt_results_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    pop_csv = pop_results_dir / "population.csv"
    manifest_csv = pop_results_dir / "population_manifest.csv"
    run_summary_csv = exec_results_dir / "run_summary.csv"
    dispatch_summary_csv = exec_results_dir / "dispatch_step_summary.csv"
    postprocess_summary_csv = post_results_dir / "postprocess_polyfit_summary.csv"
    eval_csv = sel_results_dir / "population_evaluation.csv"
    selected_csv = sel_results_dir / "selected_designs.csv"
    optimizer_eval_csv = opt_results_dir / "optimizer_evaluation.csv"
    allocation_info_json = experiment_root / "allocation_info.json"
    module_preamble_path = scripts_dir / "module_preamble.sh"

    shutil.copy2(args.population, pop_csv)
    eval_one_py = scripts_dir / eval_one_src.name
    alloc_runner_py = scripts_dir / alloc_runner_src.name
    collect_py = scripts_dir / collect_src.name
    for src, dst in [
        (eval_one_src, eval_one_py),
        (alloc_runner_src, alloc_runner_py),
        (collect_src, collect_py),
    ]:
        shutil.copy2(src, dst)
    module_preamble_path.write_text(DEFAULT_MODULE_PREAMBLE + "\n", encoding="utf-8")

    step_mm = step_for_n_points(args.start_mm, args.end_mm, args.n_points)
    expected_rows = expected_rows_for_range(args.start_mm, args.end_mm, step_mm)

    meta = {
        "project": str(project),
        "lab_root": str(lab_root),
        "run_label": run_label,
        "experiment_root": str(experiment_root),
        "execute": bool(args.execute),
        "population": str(pop_csv),
        "optimizer_evaluation": str(optimizer_eval_csv),
        "gmsh_threads": args.gmsh_threads,
        "gmsh_launcher": args.gmsh_launcher,
        "gmsh_mpi_procs": args.gmsh_mpi_procs,
        "gmsh_extra_args": args.gmsh_extra_args,
    }
    (experiment_root / "experiment_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    write_cases_cmd = f"""
python3 {write_cases_py} \
  --population {pop_csv} \
  --geo-template {geo_template} \
  --sif-template {sif_template} \
  --definition-template {def_template} \
  --outdir {cases_dir} \
  --manifest-out {manifest_csv}
"""
    print(run_shell(write_cases_cmd, cwd=project, use_modules=True, execute=args.execute))

    bind_arg = f"--bind {args.bind}" if args.bind.strip() else ""
    gmsh_extra_arg = (
        f"--gmsh-extra-args {shlex.quote(args.gmsh_extra_args)}"
        if args.gmsh_extra_args.strip()
        else ""
    )
    skip_completed_arg = "--skip-completed" if args.skip_completed else ""
    skip_first_arg = "--skip-first-node" if args.skip_first_node else ""
    fail_fast_arg = "--fail-fast" if args.fail_fast else ""
    max_individuals_arg = f"--max-individuals {args.max_individuals}" if args.max_individuals else ""
    timeout_arg = f"--timeout-sec {args.timeout_sec}" if args.timeout_sec else ""
    auto_rescue_arg = (
        f"--auto-rescue --rescue-frac {args.rescue_frac}" if args.auto_rescue else ""
    )

    dispatch_cmd = f"""
python3 {alloc_runner_py} \
  --manifest {manifest_csv} \
  --eval-helper {eval_one_py} \
  --runner {runner_mpi} \
  --project-cwd {project} \
  --out-root {runs_dir} \
  --dispatch-root {dispatch_logs_dir} \
  --summary-out {dispatch_summary_csv} \
  --allocation-info-out {allocation_info_json} \
  --module-preamble-file {module_preamble_path} \
  --label {run_label} \
  --axis {args.axis} \
  --start-mm {args.start_mm} \
  --end-mm {args.end_mm} \
  --step-mm {step_mm} \
  --expected-rows {expected_rows} \
  --nprocs {args.nprocs} \
  --inner-launcher {args.inner_launcher} \
  {bind_arg} \
  --partition-method {args.partition_method} \
  --gmsh-threads {args.gmsh_threads} \
  --gmsh-launcher {args.gmsh_launcher} \
  --gmsh-mpi-procs {args.gmsh_mpi_procs} \
  {gmsh_extra_arg} \
  --python-exe-remote {args.python_exe_remote} \
  --max-nodes {args.max_nodes} \
  {skip_first_arg} \
  {skip_completed_arg} \
  {fail_fast_arg} \
  {max_individuals_arg} \
  {timeout_arg} \
  {auto_rescue_arg}
"""
    print(run_shell(dispatch_cmd, cwd=project, use_modules=False, execute=args.execute))

    collect_cmd = f"""
python3 {collect_py} \
  --manifest {manifest_csv} \
  --runs-root {runs_dir} \
  --out {run_summary_csv} \
  --label {run_label} \
  --expected-rows {expected_rows}
"""
    print(run_shell(collect_cmd, cwd=project, use_modules=True, execute=args.execute))

    post_cmd = f"""
python3 {postprocess_polyfit_py} \
  --run-summary {run_summary_csv} \
  --polyfit-script {polyfit_py} \
  --degree 4 \
  --eval-at-mm 0.0 \
  --window-half-mm 1.0
"""
    print(run_shell(post_cmd, cwd=project, use_modules=True, execute=args.execute))

    generated_post = run_summary_csv.parent / "postprocess_polyfit_summary.csv"
    if generated_post.exists() and not postprocess_summary_csv.exists():
        shutil.copy2(generated_post, postprocess_summary_csv)

    eval_cmd = f"""
python3 {evaluate_polyfit_py} \
  --manifest {manifest_csv} \
  --postprocess-summary {postprocess_summary_csv} \
  --out {eval_csv} \
  --window-half-mm 1.0
"""
    print(run_shell(eval_cmd, cwd=project, use_modules=True, execute=args.execute))

    select_cmd = f"""
python3 {select_py} \
  --evaluation {eval_csv} \
  --out {selected_csv}
"""
    print(run_shell(select_cmd, cwd=project, use_modules=True, execute=args.execute))

    build_objectives_py = lab_root / "30_postproceso" / "build_objectives.py"
    build_cmd = f"""
python3 {build_objectives_py} \
  --population {pop_csv} \
  --evaluation {eval_csv} \
  --run-summary {run_summary_csv} \
  --postprocess-summary {postprocess_summary_csv} \
  --out {optimizer_eval_csv}
"""
    print(run_shell(build_cmd, cwd=project, use_modules=False, execute=args.execute))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
