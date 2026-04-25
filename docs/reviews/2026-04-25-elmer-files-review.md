# Revision inicial de `geo`, `sif` y `definition`

Fecha: 2026-04-25

## Resumen ejecutivo

La geometria genera una malla conforming y los IDs de cuerpos usados por `P1low.sif` coinciden con la malla actual:

| Body | Nombre esperado | Elementos |
|---:|---|---:|
| 1 | OuterMag1 | 122987 |
| 3 | OuterMag2 | 123662 |
| 5 | InnerMag1 | 86264 |
| 7 | InnerMag2 | 85843 |
| 20 | Coil | 10828 |
| 30 | Air | 618101 |

El hallazgo critico es el boundary condition externo: `Target Boundaries(3) = 32 33 34` no correspondia al exterior del aire, sino a la interfaz bobina-aire. Para la malla generada desde `StepsHTX.geo`, las fronteras exteriores 2D del aire son `36 37 38`.

## Hallazgos

1. `ExternalBC` apuntaba a la interfaz bobina-aire.

   Evidencia de ElmerGrid:

   | Boundary | Padres | Rango aproximado |
   |---:|---|---|
   | 32 | Coil-Air | radio interno bobina, `z=-0.00065..0.01205` |
   | 33 | Coil-Air | radio externo bobina, `z=-0.00065..0.01205` |
   | 34 | Coil-Air | tapa superior bobina, `z=0.01205` |
   | 36 | Air-exterior | lateral cilindro aire |
   | 37 | Air-exterior | tapa superior aire, `z=0.02205` |
   | 38 | Air-exterior | tapa inferior aire, `z=-0.02205` |

   Cambio aplicado en `simulations/elmer/base/P1low.sif`: `Target Boundaries(3) = 36 37 38`.

2. Los IDs de cuerpos funcionan en esta geometria, pero siguen siendo fragiles para barridos.

   Gmsh/ElmerGrid conservaron los tags `1,3,5,7,20,30`, asi que el caso actual calza. Para un flujo de optimizacion conviene generar `Physical Volume` y `Physical Surface` estables, o generar el `.sif` desde un preflight que lea el mapeo real de cada malla.

3. Los comentarios de magnetizacion estaban invertidos.

   `M_plus` tiene `Magnetization 3 = +8.0e5` y `M_minus` tiene `-8.0e5`. Los comentarios en los cuerpos decian lo contrario. El `.sif` del repo corrige los comentarios; la asignacion numerica no se cambio.

4. `Mesh.SaveAll = 1` hace que ElmerGrid importe tambien entidades 1D/0D como boundaries adicionales.

   No rompe el caso, pero aumenta ruido en los IDs. Cuando existan physical groups estables, conviene evaluar `Mesh.SaveAll = 0` para que la malla exporte solo lo que el solver necesita.

5. El dominio de aire parece justo para un Dirichlet `AV=0`.

   Con `air_R = 40 mm` y radio exterior de iman `25.4 mm`, el margen radial es alrededor de `14.6 mm`. Para fuerzas o sensibilidad fina, seria bueno comparar contra un dominio mas grande o una condicion de frontera abierta/infinita antes de confiar en resultados absolutos.

6. La simulacion es transiente de un solo paso con corriente constante.

   Para barridos estaticos o genetic algorithm, podria ser mas barato usar una formulacion steady-state, si el acople Circuit/CoilSolver lo permite para este caso. Si la corriente va a variar en el tiempo, entonces la estructura transiente tiene sentido.

7. `Solver 5` dice guardar VTU, pero `Exec Solver = Never`.

   Esto esta bien si la corrida del genetico solo necesita escalares y quiere ahorrar I/O. Para diagnostico visual, conviene activar un `ResultOutputSolver` en una variante de debug, no en todas las corridas.

8. `HMB_circuit.definition` esta razonablemente consistente con la geometria de bobina.

   `Ae_Coil1 = 7.62e-5` coincide con `(rb2-rb1)*hb = 0.006*0.0127`. Para optimizacion, esa relacion deberia salir de una sola fuente de parametros y no quedar duplicada entre `.geo` y `.definition`.

## Recomendaciones proximas

- Agregar physical groups en el `.geo` o generar automaticamente los IDs del `.sif` desde el reporte de preflight.
- Crear niveles de malla (`coarse`, `normal`, `fine`) para no correr el genetic algorithm con malla de validacion fina.
- Separar casos `HMB` y `PMB-only`: `use_coil = 0` en `.geo` no es compatible con el `.sif` actual porque sigue esperando cuerpo de bobina, circuito y `Body Force 1`.
- Centralizar parametros compartidos (`rb1`, `rb2`, `hb`, `N`, `R`, `I`, magnetizacion) en YAML/JSON o en un generador unico.
