# Informe F7: eficiencia e inferencia

## 1. La puerta

> Mejora medida de tokens/s o memoria con la misma calidad, documentada.

Hay dos mejoras medidas: **tokens/s con KV cache** y **memoria con INT8**. Las dos se comparan contra
el baseline del mismo modelo de 1.401.792 parámetros.

## 2. Baseline

Generación de 48 tokens desde `"En un lugar de la Mancha"`, temperatura 0.8, semilla 7, tres corridas:

| Modo | Tiempo medio | Tokens/s |
| --- | ---: | ---: |
| Sin caché | 5924 ms | **8.10** |

Cada token vuelve a procesar toda la secuencia: $O(L^2)$ por token.

## 3. KV cache: 25.18×

| Modo | Tiempo medio | Tokens/s |
| --- | ---: | ---: |
| Sin caché | 5736 ms | 8.34 |
| **Con KV cache** | **222 ms** | **210.06** |

- **25.18×** más rápido.
- **Salida idéntica** token a token.
- Se guardan K/V por capa y cabeza; el token nuevo se atiende contra el pasado con RoPE por posición.
- El precio es memoria $O(L)$ por capa/cabeza.

## 4. Cuantización INT8: 3.93× de memoria

| Formato | Bytes | Ratio | Perplejidad |
| --- | ---: | ---: | ---: |
| Float32 | 5,35 MB | 1,00× | **379.30** |
| **INT8 por canal** | **1,36 MB** | **3,93×** | 379.34 |
| INT8 por tensor | 1,34 MB | 4,00× | 510.16 |

- Per-channel ahorra **3,93×** con perplejidad prácticamente igual (+0,04).
- Per-tensor ahorra 4,00× pero degrada a 510.16: la escala única no alcanza.
- Es memoria de **pesos almacenados**; no hay matmul INT8 ni mejora de tokens/s.

## 5. Sampling

Se añadieron temperatura, top-k y top-p. Greedy degenera en bucles de `a`/`que`; más temperatura da
variedad, no calidad. Sampling no cambia tokens/s: el coste está en el modelo.

## 6. Reproducción

```
node lab/transformer/benchmark.ts
node lab/transformer/benchmark-cache.ts
node lab/transformer/sampling.ts
node lab/transformer/quantize.ts
```

- [inference-baseline.json](../../data/inference-baseline.json)
- [inference-kv-cache.json](../../data/inference-kv-cache.json)
- [sampling-report.json](../../data/sampling-report.json)
- [quantization-report.json](../../data/quantization-report.json)

## 7. Límites

- Node/CPU, una secuencia, sin batch ni beam search.
- La caché crece con la longitud; el prefill del prompt sigue siendo $O(P^2)$.
- La cuantización se midió como almacenamiento; para velocidad haría falta una matmul INT8.
- El modelo base es el de F6, con perplejidad alta; las mejoras son de inferencia, no de calidad.

## 8. Conclusión

La puerta se cumple con evidencia: **25.18× en tokens/s** manteniendo la salida idéntica, y **3.93×
menos memoria de pesos** con perplejidad casi igual. La calidad no mejoró; la eficiencia sí, y está
medida antes y después.
