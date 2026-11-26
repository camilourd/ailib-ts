# Sets y benchmark

## 1. Qué problema resuelve

«¿Qué modelo es mejor?» no tiene respuesta sin un **set**: los mismos datos, el mismo split, la misma
métrica y un baseline. Si cada experimento usa su propio trozo de datos o su propia métrica, las
comparaciones no significan nada.

Un **set** empaqueta una tarea: datos, separación temporal, pérdida, métrica y baseline. El
**benchmark** entrena varios modelos sobre el mismo set y devuelve una tabla comparable.

## 2. Definición y notación

- **Split**: `input` y `target`, dos tensores con las mismas filas.
- **Set**: `train`, `validation` y `test`; su `loss` de entrenamiento, su `metric` de evaluación y un
  `baseline` de referencia.
- **Model**: algo con `forward(input)` y `parameters()` (para el optimizador).
- **Candidato**: un modelo con su optimizador y su número de pasos.
- **Benchmark**: entrena cada candidato en `train`, lo evalúa en `test` y devuelve
  `{ nombre, métrica, valor, baseline }`.

Los sets viven en carpetas por dominio:

```
src/sets/
├── types.ts, metrics.ts, runner.ts
├── tabular/synthetic-asset.ts
├── images/   (futuro)
└── text/     (futuro)
```

## 3. Por qué así

- **Split cronológico**: en series temporales no se baraja; el test tiene que ser **posterior** al
  train. Barajar mezcla futuro con pasado y el resultado es una mentira.
- **Normalización con el train**: la media y la desviación se calculan solo con train y se aplican a
  validación y test. Calcularlas con todo el dataset es fuga de información.
- **Baseline**: sin una referencia (predecir 0, el último valor, la clase mayoritaria) no se sabe si el
  modelo aporta algo.
- **Misma pérdida y misma métrica para todos**: el runner no deja que cada modelo se evalúe a su
  manera.

## 4. El primer set: activo sintético

- Un activo con movimiento browniano geométrico y un segundo activo correlacionado.
- Tres variables conocidas en $t$: retorno del día, volatilidad móvil de 10 días y retorno del segundo
  activo.
- Objetivo: retorno del día siguiente en porcentaje, $r_{t+1}\cdot100$.
- Split 70/15/15 por orden temporal; normalización con el train; baseline: predecir 0 (MAE de los
  retornos absolutos del test).

Con 800 días y semilla 7, el primer benchmark da:

| Modelo | MAE |
| --- | --- |
| Baseline (0) | 0.7946 |
| Lineal | 0.8039 |
| MLP `3 → 8 → 1` | 0.7924 |

La lección: **predecir retornos es difícil**; el MLP apenas roza al baseline. Un resultado así es
valioso: evita creer que un modelo «funciona» solo porque tiene capas.

## 5. Implementación

- [`src/sets/types.ts`](../../src/sets/types.ts): `Split`, `RegressionSet`, `Candidate`, `Result`.
- [`src/sets/metrics.ts`](../../src/sets/metrics.ts): MAE, RMSE y acierto direccional.
- [`src/sets/runner.ts`](../../src/sets/runner.ts): `benchmark` entrena y evalúa cada candidato.
- [`src/sets/tabular/synthetic-asset.ts`](../../src/sets/tabular/synthetic-asset.ts): el set sintético.
- [`src/nn/model.ts`](../../src/nn/model.ts) y [`src/nn/mlp.ts`](../../src/nn/mlp.ts): el contrato de
  modelo y el MLP denso.
- [`lab/sets/run.ts`](../../lab/sets/run.ts): el benchmark con lineal y MLP.

## 6. Verificación

- El set tiene splits 132/28/29 para 200 días y formas `[n, 3]` / `[n, 1]`.
- Las features del train quedan con media 0 y desviación 1; la misma semilla da los mismos datos.
- MAE, RMSE y acierto direccional coinciden con el cálculo a mano.
- El benchmark entrena cada candidato y devuelve valores finitos con el baseline del set.
- 93/93 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Un solo set de regresión**; clasificación e imágenes/texto llegarán con su propio `Split`.
- **Sin batching ni barajado**: full-batch, como el resto de F2.
- **Sin validación en el runner**: el split existe, pero aún no se usa para *early stopping*.
- **Datos sintéticos**: el resultado es limpio y reproducible, pero no prueba que funcione con datos
  reales.
- **MAE sola**: el acierto direccional y el $R^2$ se calcularán en las piezas de pronóstico.
- **`f32` y CPU.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 5 (machine learning básico).
- scikit-learn, *Train/test split* y *TimeSeriesSplit* (por qué no barajar en series temporales).
- Gu, Kelly y Xiu (2020), *Empirical Asset Pricing via Machine Learning* (benchmarks y errores).
