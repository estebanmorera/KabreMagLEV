# Caso base Elmer/Gmsh

Archivos importados y normalizados:

- `StepsHTX.geo`
- `P1low.sif`
- `HMB_circuit.definition`

El `.sif` ya incluye la correccion de `ExternalBC` validada por preflight para esta geometria: `36 37 38` son las superficies exteriores 2D del dominio de aire.

Antes de cambiar dimensiones o activar/desactivar bobina, correr:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/preflight_elmer_mesh.ps1
```
