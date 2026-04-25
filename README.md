# Proyecto VAD

Repositorio de trabajo para ordenar simulaciones, notebooks y scripts alrededor del VAD/HMB/PMB con ElmerFEM y Gmsh.

## Estructura

- `simulations/elmer/base`: caso base importado desde los archivos actuales (`StepsHTX.geo`, `P1low.sif`, `HMB_circuit.definition`).
- `simulations/elmer/generated`: mallas, reportes y salidas reproducibles generadas por scripts. No se versiona.
- `notebooks`: notebooks separados por enfoque para evitar que el genetico, la paralelizacion y las pruebas de `.geo/.sif` se mezclen.
- `scripts`: automatizacion de preflight, generacion de casos y validaciones.
- `data/raw`: datos de entrada externos o congelados.
- `data/processed`: tablas intermedias reproducibles.
- `data/runs`: resultados de corridas y experimentos.
- `docs`: revisiones tecnicas y decisiones de estructura.

## Estado inicial

Los tres archivos importados quedaron con nombres canonicos para que el `.sif` encuentre el `.definition` sin depender de sufijos de descarga:

- `simulations/elmer/base/StepsHTX.geo`
- `simulations/elmer/base/P1low.sif`
- `simulations/elmer/base/HMB_circuit.definition`

El primer preflight con Gmsh + ElmerGrid confirmo que los cuerpos volumetricos usados por el `.sif` coinciden para esta geometria, pero tambien detecto que el `ExternalBC` original apuntaba a la interfaz bobina-aire. El `.sif` del repo ya usa las fronteras exteriores del aire para esta malla.

Ver `docs/reviews/2026-04-25-elmer-files-review.md`.
