# Pronóstico de retornos

## 1. Qué problema resuelve

Se quiere pronosticar un activo. El precio no es un buen objetivo: no es estacionario y «mañana valdrá
lo de hoy» ya da un error pequeño. El objetivo es el **retorno** del día siguiente,
$r_{t+1} = \ln(C_{t+1}/C_t)$, y la pregunta útil es si un modelo aporta algo frente a un baseline.

## 2. Definición y notación

- **Features** en $t$: retorno del día, volatilidad móvil de 10 días y retorno de un segundo activo
  correlacionado.
- **Objetivo**: retorno del día siguiente en porcentaje, $100\cdot r_{t+1}$.
- **Split** cronológico 70/15/15; normalización calculada solo con train.
- **Baseline**: predecir 0 (retorno nulo).
- **Métricas de error**: MAE, RMSE, R², sesgo y correlación.
- **Métricas de dirección**: matriz de confusión, accuracy, precisión, recall, especificidad, F1,
  balanced accuracy, MCC y tasas de «sube».

## 3. Por qué así

- **Contra el baseline**: un modelo solo sirve si mejora una referencia trivial. En finanzas, el
  baseline suele ser muy difícil de batir.
- **Contra el azar en dirección**: acertar el signo es otra pregunta; un modelo puede tener buen MAE y
  no acertar la dirección, o al contrario.
- **Sin fuga**: features de $t$, objetivo de $t+1$; normalización solo train.
- **Datos sintéticos**: el GBM genera retornos independientes, así que **no hay señal predecible**. Si
  un modelo batiera claramente al baseline, habría que sospechar de un error o de una fuga.

## 4. Resultado con 1200 días y semilla 7

| Modelo | MAE | RMSE | R² | Sesgo | Corr |
| --- | --- | --- | --- | --- | --- |
| Baseline (0) | 0.8230 | 1.0214 | −0.0087 | −0.0949 | 0.0000 |
| Lineal | 0.8256 | 1.0241 | −0.0139 | −0.1026 | −0.0219 |

| Modelo | Acc | Prec | Recall | Espec | F1 | Balanced | MCC | Sube pred. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.49 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.00 |
| Lineal | 0.44 | 0.45 | 0.42 | 0.46 | 0.44 | 0.44 | −0.12 | 0.48 |

La tasa real de días que suben es 0.51. El baseline siempre dice «baja» (acierta los días de baja y
falla todos los de subida); la lineal reparte sus predicciones pero no acierta más que el azar: su MCC
es −0.12. El MAE por dirección es 0.9018 en días de subida y 0.7451 en días de bajada.

Conclusión: **en un GBM no hay señal que aprender**. Que el modelo no bata al baseline es el resultado
correcto; el valor del experimento es el protocolo y las métricas, no el número.

### Comparación con validación y MLP

Para comparar de verdad, elegí hiperparámetros con **validación** y dejé el test para el final: tasa de
aprendizaje para la lineal; tamaño oculto y tasa para el MLP. Entrené en train, elegí por MAE de
validación y reentrené al ganador con train + validación.

La validación eligió `lr = 0.5` para la lineal (MAE 0.7848) y `8` ocultas con `lr = 0.01` para el MLP
(MAE 0.7834). En test:

| Modelo | MAE | RMSE | R² | MCC | Acc. dirección |
| --- | --- | --- | --- | --- | --- |
| Baseline (0) | 0.8230 | 1.0214 | −0.0087 | 0.00 | 0.49 |
| Lineal `lr=0.5` | 0.8258 | 1.0249 | −0.0155 | −0.06 | 0.47 |
| MLP `8, lr=0.01` | 0.8308 | 1.0304 | −0.0264 | −0.11 | 0.45 |

La validación prometía una mejora (0.7848/0.7834), pero **el test no la confirma**: ninguno supera el
baseline. Con 1200 días, el test es otro tramo de ruido; elegir por validación no garantiza
generalizar. La conclusión no cambia: en un GBM sin señal, el MLP tampoco aporta.

## 5. Implementación

- [`lab/forecast/linear.ts`](../../lab/forecast/linear.ts): entrena `Linear(3, 1)` con MSE y SGD,
  evalúa en test y compara error y dirección contra el baseline.
- [`lab/forecast/compare.ts`](../../lab/forecast/compare.ts): lineal y MLP con selección por
  validación; test solo al final.
- [`src/sets/tabular/synthetic-asset.ts`](../../src/sets/tabular/synthetic-asset.ts): genera el set.
- [`src/sets/direction.ts`](../../src/sets/direction.ts): matriz y métricas de dirección.
- [`src/sets/metrics.ts`](../../src/sets/metrics.ts): MAE, RMSE, R², sesgo y correlación.

## 6. Verificación

- El set mantiene el orden temporal y normaliza solo con train (probado en `test/sets/`).
- El laboratorio imprime las dos tablas y una cobertura de 1.00 (no hay ceros reales en el test).
- El laboratorio de comparación imprime la selección por validación y las tablas de test; el MLP no
  supera al baseline.
- Las métricas de dirección están probadas con matrices a mano (`test/sets/direction.test.ts`).
- 101/101 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Datos sintéticos**: el GBM no tiene autocorrelación; no prueba que haya señal en un activo real.
- **Un solo split de validación**: la selección puede sobreajustar la validación; el test es el que
  manda. Sin *walk-forward*.
- **Sin costes de transacción ni estrategia**: medir el signo no es ganar dinero.
- **Un solo activo y tres features**: sin ingeniería de features ni datos de mercado.
- **Ceros excluidos**: si hubiera muchos retornos 0, la cobertura de dirección bajaría.
- **`f32`.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Gu, Kelly y Xiu (2020), *Empirical Asset Pricing via Machine Learning*.
- Lo (2002), *The Statistics of Sharpe Ratios* (por qué el ruido importa en finanzas).
- Documentación de `TimeSeriesSplit` de scikit-learn.
