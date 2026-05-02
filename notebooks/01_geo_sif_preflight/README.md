# 01_geo_sif_preflight

Notebook para revisar geometria, malla, IDs de cuerpos/fronteras y consistencia entre `.geo`, `.sif` y `.definition`.

- `geo_sif_mesh_design_trials.ipynb`: compara la geometria anular actual contra una variante con imanes internos tipo pastilla, genera casos `.geo/.sif/.definition`, corre preflight Gmsh/ElmerGrid, ajusta `ExternalBC` y grafica metricas de malla/resultados.
- `requirements_geo_sif_trials.txt`: dependencias Python sugeridas para el kernel del notebook.
