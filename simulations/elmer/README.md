# Simulaciones Elmer/Gmsh

- `base`: archivos fuente mantenidos a mano.
- `generated`: mallas, reportes y casos generados por scripts. Esta carpeta no se versiona salvo `.gitkeep`.

Flujo recomendado:

1. Editar o generar un `.geo`.
2. Ejecutar `scripts/preflight_elmer_mesh.ps1` para producir malla y reporte de IDs.
3. Ajustar/generar el `.sif` con cuerpos y fronteras validadas.
4. Correr ElmerSolver solo despues de que el preflight confirme el mapeo.
