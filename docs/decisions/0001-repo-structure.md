# Decision 0001: estructura de trabajo VAD

Fecha: 2026-04-25

## Decision

Separar el proyecto en cuatro carriles principales:

- `simulations`: archivos fuente de Elmer/Gmsh y casos generados.
- `notebooks`: exploracion por enfoque, no por archivo suelto.
- `scripts`: pasos repetibles que no deberian vivir solo en notebooks.
- `data`: entradas, intermedios y resultados de corridas.

## Razonamiento

El proyecto mezcla geometria parametrica, archivos `.sif`, definiciones de circuito, barridos, optimizacion y genetic algorithm. Si todo vive en notebooks planos, se vuelve dificil saber que es fuente, que es resultado y que fue una prueba descartable.

La regla propuesta es:

- El notebook explora y explica.
- El script ejecuta algo repetible.
- `simulations/elmer/base` guarda el caso base legible.
- `simulations/elmer/generated` guarda artefactos regenerables.
- `data/runs` guarda resultados de experimentos con fecha/configuracion.

## Notas

No recupere automaticamente la estructura exacta mencionada de los otros chats. Esta base queda pensada para adaptarse rapido si Kabre o los otros hilos ya tienen nombres de columnas, schemas o convenciones especificas.
