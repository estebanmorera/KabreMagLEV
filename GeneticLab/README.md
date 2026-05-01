# GeneticLab

Plantilla para probar optimizadores evolutivos y surrogate-assisted sobre el
flujo existente de `genes`.

Ruta prevista en el cluster:

```text
/work/jmorera/Genes/
  genes/
  Genetic1/
  minicycle/
  testparallel_alloc/
  GeneticLab/
```

## Estructura

- `00_config`: espacio de diseno y aliases de objetivos.
- `10_poblacion`: generacion/reparacion de individuos.
- `20_ejecucion`: wrapper para llamar al pipeline existente de `genes`.
- `30_postproceso`: conversion de resultados a objetivos/constraints.
- `40_optimizacion`: ask/tell para algoritmos tipo MOEA.
- `notebooks`: notebook guia para las pruebas iniciales.

## Idea de uso

1. Crear un ambiente Python con `numpy`, `pandas`, `matplotlib` y `pymoo`.
2. Abrir `notebooks/genes_moea_initial_trials.ipynb`.
3. Elegir algoritmo (`age2`, `rvea`, `ctaea`, `sms`, `nsga3`, `random`).
4. Generar una poblacion candidata con `ask`.
5. Correr el pipeline Elmer/Gmsh existente.
6. Convertir resultados a `optimizer_evaluation.csv`.
7. Hacer `tell` y repetir.

Esta carpeta no reemplaza `genes`; lo usa como backend de simulacion.
