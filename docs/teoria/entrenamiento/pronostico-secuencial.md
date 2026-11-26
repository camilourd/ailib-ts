# Pronóstico secuencial de activos

## 1. Qué problema resuelve

El primer experimento era **tabular**: tres features del día y el retorno del día siguiente. Aquí se
prueba si las arquitecturas de secuencia —RNN, GRU, LSTM y transformer— encuentran algo que el modelo
tabular no vio. No se cambia el objetivo: sigue siendo el retorno del día siguiente.

## 2. Datos y protocolo

- Mismo generador GBM del primer set, semilla 7: activo 1, activo 2 correlacionado y volatilidad móvil.
- Ventana de **10 días**; cada día aporta `[retorno1, volatilidad móvil, retorno2]`.
- Objetivo: retorno del activo 1 al día siguiente, en porcentaje.
- Split cronológico 70/15/15: **832 train / 178 validación / 179 test**.
- Normalización calculada **solo con train**.
- 1000 pasos de SGD, `lr=0.005`, clipping 1.0, 3 semillas por modelo.
- Sin ajuste de hiperparámetros por modelo: la comparación es a configuración fija.

## 3. Modelos

| Modelo | Arquitectura |
| --- | --- |
| RNN | Elman, oculto 16; último estado → `Linear(16,1)` |
| GRU | oculto 16; último estado → `Linear(16,1)` |
| LSTM | oculto 16; último estado → `Linear(16,1)` |
| Transformer | `Linear(3→32)`, 2 bloques de 4 cabezas, FFN 64, RoPE causal; última posición → `Linear(32,1)` |

## 4. Resultados

| Modelo | Parámetros | MAE (media ± sd) | RMSE | R² | Accuracy | MCC | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cero (baseline) | 0 | **0.8147** ± 0.0000 | **1.0107** | **−0.0105** | 0.49 | 0.00 | 0 | 0 | 87 | 92 |
| RNN | 337 | 0.8333 ± 0.0116 | 1.0489 | −0.0882 | 0.51 | 0.02 | 134 | 121 | 140 | 142 |
| GRU | 977 | 0.8282 ± 0.0077 | 1.0286 | −0.0466 | 0.49 | −0.02 | 112 | 111 | 150 | 164 |
| LSTM | 1297 | 0.8207 ± 0.0037 | 1.0202 | −0.0296 | 0.51 | 0.03 | 116 | 101 | 160 | 160 |
| Transformer | 17.249 | 0.9106 ± 0.0175 | 1.1508 | −0.3105 | 0.54 | 0.08 | 129 | 102 | 159 | 147 |

La matriz del baseline es de un test (179 días); las de los modelos suman los tres seeds (537 días).

## 5. Lectura

- **Ningún modelo supera al baseline de predecir 0 en MAE.** El mejor es el LSTM (0.8207), todavía por
  encima de 0.8147.
- Todos los **R² son negativos**: no baten a predecir la media.
- La accuracy direccional está entre 0.49 y 0.54, y el **MCC entre −0.02 y 0.08**: no hay ventaja
  direccional real.
- El transformer es el peor en MAE y R² y el que más parámetros tiene: sobreajusta un GBM sin señal.
- El resultado es el esperado: un **GBM no tiene retorno predecible**. Si un modelo superara mucho al
  baseline, lo primero sería buscar una fuga, no celebrar.

## 6. Matriz de confusión

La clase positiva es «sube». El baseline predice siempre 0, que cuenta como «baja»: por eso tiene
`TP=0` y `FN=92` (todos los días que subieron). Los modelos se reparten los errores casi por igual;
ninguno tiene precisión o recall útiles. La matriz completa está en `data/forecast-sequence.json`.

## 7. Implementación

- [`lab/forecast/sequence.ts`](../../lab/forecast/sequence.ts): datos, modelos, entrenamiento,
  métricas y matriz.
- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `sliceRows` para la última posición del
  transformer.
- [`test/core/slices.test.ts`](../../test/core/slices.test.ts): gradiente de `sliceRows`.
- [`data/forecast-sequence.json`](../../data/forecast-sequence.json): resultados por modelo.

## 8. Límites y errores comunes

- **Un solo activo sintético y una sola semilla de datos**: no generaliza a activos reales.
- **Sin tuning por modelo**: la comparación es justa en presupuesto, no en esfuerzo.
- **El transformer causal usa 10 posiciones**: no es un transformer grande.
- **Accuracy alta con MCC bajo no es edge**: mirar siempre la matriz y el balance.
- **Un GBM no tiene señal**: el experimento mide el arnés, no una estrategia rentable.
- **No se incluyen costes, slippage ni tamaño de posición.**

## 9. Para profundizar (opcional)

- Gu, Kelly y Xiu (2020), *Empirical Asset Pricing via Machine Learning*.
- Fischer y Krauss (2018), *Deep learning with long short-term memory networks for financial market predictions*.
- Documentación de métricas direccionales en `teoria/entrenamiento/direccion.md`.
