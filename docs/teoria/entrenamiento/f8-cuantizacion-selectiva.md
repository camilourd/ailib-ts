# Cuantización selectiva por módulos

## 1. Pregunta propia

¿Qué módulos del transformer toleran INT8 sin perder perplejidad? La intuición habitual dice que
atención y FFN aguantan, mientras embedding, salida y normas son sensibles. Este experimento lo mide
en el modelo de F6.

## 2. Hipótesis

> Atención y FFN toleran INT8; embedding/salida/normas concentran la degradación.

Si la hipótesis fuera cierta, cuantizar solo atención/FFN daría casi todo el ahorro con menos pérdida
que cuantizar todo.

## 3. Protocolo

- Modelo de 1.401.792 parámetros, checkpoint del paso 1000.
- 8 ventanas de validación de 128 tokens; perplejidad = $e^{\text{pérdida media}}$.
- Cuantización INT8 **simétrica por canal** (una escala por columna); normas y sesgos, por tensor.
- Grupos:
  - `embedding`;
  - `attention`: 4 proyecciones por bloque;
  - `ffn`: dos capas por bloque;
  - `norms`: LayerNorm;
  - `output`.
- 9 variantes: float32, todo, cada grupo solo, y todo menos un grupo.
- Se restaura el modelo float32 entre variantes.

## 4. Resultados

| Variante | MB | Ratio | Perplejidad |
| --- | ---: | ---: | ---: |
| float32 | 5,35 | 1,00× | **379.30** |
| Todo INT8 | **1,36** | **3,93×** | 379.34 |
| Solo atención | 4,18 | 1,28× | 379.38 |
| Solo FFN | 3,01 | 1,78× | 379.29 |
| Solo embedding+salida | 4,88 | 1,10× | 379.34 |
| Todo menos atención | 2,53 | 2,11× | 379.26 |
| Todo menos FFN | 3,70 | 1,44× | 379.35 |
| Todo menos embedding/salida | 1,83 | 2,92× | **379.30** |
| Todo menos normas | 1,37 | 3,90× | 379.41 |

## 5. Lectura

- **La hipótesis no se cumple**: con escala por canal, **ningún módulo es claramente sensible**. Todas
  las variantes quedan entre 379.26 y 379.41.
- El ahorro no requiere selección: **todo INT8** da 3,93× con +0,04 de perplejidad.
- La diferencia grande de F7 era **per-tensor vs per-channel**, no el módulo. Una sola escala global
  destruía la calidad; una por columna la conserva.
- `todo menos embedding/salida` consigue 2,92× con la misma perplejidad que float32, pero ahorra menos
  que cuantizar todo.
- Para este modelo, **cuantizar todo por canal es la mejor relación memoria/calidad**.

## 6. Reproducción

```
node lab/transformer/selective-quantization.ts
```

- [`lab/transformer/selective-quantization.ts`](../../lab/transformer/selective-quantization.ts)
- [`data/selective-quantization.json`](../../data/selective-quantization.json)
- Requiere `checkpoints/seed-1-step-1000.json` (no versionado).

## 7. Implementación

- [`src/nn/quantization.ts`](../../src/nn/quantization.ts): `quantizeInt8`/`dequantizeInt8`.
- [`lab/transformer/quantize.ts`](../../lab/transformer/quantize.ts): cuantización completa.
- `groupOf` en el laboratorio clasifica los nombres de `namedParameters` por posición.

## 8. Límites

- Un modelo, un checkpoint y 8 ventanas: el resultado no es una ley general.
- Solo se mide perplejidad; no hay evaluación de generación ni de tareas.
- La cuantización es post-entrenamiento, sin calibración.
- La memoria es de pesos almacenados; no hay matmul INT8.
- El modelo base tiene perplejidad alta; la conclusión vale para este presupuesto.

## 9. Conclusión

La pregunta era «¿qué módulos toleran INT8?». La respuesta medida es: **todos, si la escala es por
canal**. La sensibilidad que se atribuye a embedding o salida no aparece aquí; el factor decisivo es
la granularidad de la escala. El experimento deja una decisión práctica: cuantizar todo por canal.
