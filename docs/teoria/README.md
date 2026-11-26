# Índice de teoría

Mapa entre conceptos, fases de la hoja de ruta y código. Se actualiza en el mismo cambio que la nota
o el código; el método está en [`../metodo.md`](../metodo.md).

| Concepto | Fase | Nota | Código | Test | Estado |
| --- | --- | --- | --- | --- | --- |
| sigmoid | F0 (ejemplo) · F2 | [sigmoid.md](redes/sigmoid.md) | `src/nn/sigmoid.ts` | `test/nn/sigmoid.test.ts` | borrador |
| rng | F1 | [rng.md](nucleo/rng.md) | `src/core/random/` | `test/core/random/rng.test.ts` | borrador |
| tensor | F1 | [tensor.md](nucleo/tensor.md) | `src/core/tensor.ts` | `test/core/tensor.test.ts` | borrador |
| broadcasting | F1 | [broadcasting.md](nucleo/broadcasting.md) | `src/core/tensor-ops.ts` | `test/core/tensor-ops.test.ts` | borrador |
| matmul | F1 | [matmul.md](nucleo/matmul.md) | `src/core/tensor-ops.ts` | `test/core/matmul.test.ts` | borrador |
| reducciones | F1 | [reductions.md](nucleo/reductions.md) | `src/core/tensor-ops.ts` | `test/core/reductions.test.ts` | borrador |
| autograd | F1 | [autograd.md](nucleo/autograd.md) | `src/core/tensor.ts`<br>`src/core/tensor-ops.ts` | `test/core/autograd.test.ts` | borrador |
| alternativas a backprop | F3/F8 | [alternativas.md](entrenamiento/alternativas.md) | — | — | mapa |
| SGD | F2 | [sgd.md](entrenamiento/sgd.md) | `src/optim/sgd.ts` | `test/optim/sgd.test.ts` | borrador |
| capa lineal | F2 | [capa-lineal.md](redes/capa-lineal.md) | `src/nn/linear.ts` | `test/nn/linear.test.ts` | borrador |
| relu | F2 | [relu.md](redes/relu.md) | `src/core/tensor-ops.ts` | `test/core/unary-ops.test.ts` | borrador |
| exp/log | F2 | [exp-log.md](nucleo/exp-log.md) | `src/core/tensor-ops.ts` | `test/core/unary-ops.test.ts` | borrador |
| pérdidas | F2 | [perdidas.md](entrenamiento/perdidas.md) | `src/nn/losses.ts` | `test/nn/losses.test.ts` | borrador |
| bucle de entrenamiento | F2 | [bucle.md](entrenamiento/bucle.md) | `src/train/trainer.ts` | `test/train/trainer.test.ts` | borrador |
| checkpoints | F2 | [checkpoints.md](entrenamiento/checkpoints.md) | `src/io/checkpoint.ts` | `test/io/checkpoint.test.ts` | borrador |
| sets y benchmark | F2 | [sets.md](entrenamiento/sets.md) | `src/sets/` | `test/sets/` | borrador |
| dirección | F2 | [direccion.md](entrenamiento/direccion.md) | `src/sets/direction.ts` | `test/sets/direction.test.ts` | borrador |
| pronóstico de retornos | F2 | [pronostico.md](entrenamiento/pronostico.md) | `lab/forecast/compare.ts` | — | borrador |
| HAEA | F3 | [haea.md](evolucion/haea.md) | `src/evolution/` | `test/evolution/haea.test.ts` | borrador |
| MLP | F2 | [mlp.md](redes/mlp.md) | `src/nn/mlp.ts` | `test/nn/mlp.test.ts` | borrador |

Reglas:

- Una fila por concepto; el enlace de **Nota** es la nota canónica.
- **Fase** es la de la hoja `ailib-ts` en la capa (`F0`…`F8`).
- **Estado**: `pendiente` (concepto adoptado, sin nota todavía), `borrador`, `estable` o `mapa`
  (nota de contexto sin código todavía).
- **Código** y **Test** pueden tener varios archivos; sepáralos con `<br>`.
- Los conceptos que todavía no se adoptan no entran aquí: viven en
  [`../inventario-csharp.md`](../inventario-csharp.md).
