# Notebooks

Propuesta de orden para trabajar sin mezclar enfoques:

- `00_project_index`: mapa del proyecto, convenciones de nombres y resumen de resultados confiables.
- `01_geo_sif_preflight`: inspeccion de `.geo`, `.sif`, IDs de cuerpos/fronteras, pruebas de malla y sanity checks.
- `02_parallel_runs`: pruebas de paralelizacion, colas locales, locks de carpetas de salida y medicion de tiempos.
- `03_genetic_algorithm`: corrida del genetico, poblaciones, checkpoints y analisis de convergencia.
- `04_single_script_optimization`: micro-optimizaciones de scripts individuales antes de subirlas al flujo completo.

Regla practica: cuando una celda empieza a ser necesaria para repetir una corrida, moverla a `scripts/` y dejar el notebook como orquestador o analisis.
