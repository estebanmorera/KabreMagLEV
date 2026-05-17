# 01_geo_sif_preflight

Notebook para revisar geometria, malla, IDs de cuerpos/fronteras y consistencia entre `.geo`, `.sif` y `.definition`.

- `geo_sif_mesh_design_trials.ipynb`: guia para Kabre (`/work/jmorera/Genes`) que compara la geometria anular actual contra variantes con imanes internos tipo pastilla.
  - Incluye escenarios PMB tipo Ruben, pastilla limitada por radio real menos pared, y pastilla de area equivalente como diagnostico.
  - Genera pares `I_eval_A = 0` y `I_eval_A = 3` para medir cuanto se ve el AMB sobre cada PMB.
  - Mantiene capas externas de malla fijas y concentra cambios de refinamiento en rotor interno y bobina.
  - Corre preflight Gmsh/ElmerGrid para ajustar `ExternalBC` por escenario y exporta reportes comparables en `reports/`.
  - El SIF candidato limpia el solver inactivo 5 y mantiene `Use Piola Transform = Logical True`; tambien incluye una prueba opcional con Piola apagado para sensibilidad.
  - Permite probar mallado paralelo de Gmsh en Kabre con `GMSH_THREADS` (`gmsh -nt`) y, de forma experimental, `GMSH_MPI_PROCS`; exporta `mesh_timing_summary.csv`.
  - El notebook conserva `RUN_LABEL` al re-ejecutar la configuracion para no mezclar templates, manifests, cases y runs entre carpetas distintas.
- `requirements_geo_sif_trials.txt`: dependencias Python sugeridas para el kernel del notebook.
