# Gmsh parallel meshing on Kabre

Contexto del modulo recibido de Kabre:

- `module load gmsh/4.15.0` carga tambien `OpenCASCADE/7.9.3` y `mpich/4.1.1`.
- Este build debe correr en nodos Nukwa por compatibilidad de GLIBC.
- La documentacion upstream de Gmsh mantiene MPI como experimental y no usado directamente para meshing, por lo que la primera prueba debe ser OpenMP con `gmsh -nt N`.

Implementacion en el repo:

- `genes/20_ejecucion/sweep_mpi.py` acepta `--gmsh-threads`, `--gmsh-launcher`, `--gmsh-mpi-procs` y `--gmsh-extra-args`.
- Por defecto usa Gmsh serial con `-nt N`: `--gmsh-launcher serial --gmsh-mpi-procs 1 --gmsh-threads 4`.
- Si se quiere probar MPI explicito en Nukwa:
  `--gmsh-launcher mpirun --gmsh-mpi-procs 2 --gmsh-threads 1`.
- Cada corrida guarda en `diag_sweep_results.csv`: `gmsh_threads`, `gmsh_mpi_procs`, `gmsh_launcher`, `gmsh_elapsed_raw` y `gmsh_command`.
- El notebook `notebooks/01_geo_sif_preflight/geo_sif_mesh_design_trials.ipynb` exporta `reports/mesh_timing_summary.csv` para comparar tiempos de mallado.

Recomendacion inicial:

1. Correr el mismo caso con `GMSH_THREADS=1`, `4` y `8`.
2. Comparar `mesh_timing_summary.csv` y confirmar que el numero de elementos no cambia de forma relevante.
3. Probar MPI solo despues: `GMSH_LAUNCHER=mpirun`, `GMSH_MPI_PROCS=2`, `GMSH_THREADS=1`.
4. Mantener `GMSH_MPI_PROCS=1` si MPI no reduce tiempo o genera logs/salidas inconsistentes.
