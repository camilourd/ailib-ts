# Dirección

## 1. Qué problema resuelve

Predecir el retorno tiene dos preguntas distintas: **cuánto** (magnitud) y **hacia dónde** (signo). La
MAE mide la magnitud, pero un modelo puede tener un error pequeño y equivocarse casi siempre de signo.
La dirección necesita su propia matriz de confusión y sus métricas.

## 2. Definición y notación

- Clase positiva: **«sube»**, retorno > 0.
- Clase negativa: **«baja»**, retorno < 0.
- Un retorno real exactamente 0 no tiene dirección: se cuenta como `ties` y queda **fuera** de la
  matriz.
- Una predicción 0 cuenta como «baja» (`> 0` es la frontera).

Matriz con filas = real, columnas = predicho:

|  | predicho sube | predicho baja |
| --- | --- | --- |
| **real sube** | TP | FN |
| **real baja** | FP | TN |

## 3. Métricas y por qué

- **Accuracy** $= \frac{TP+TN}{N}$: aciertos totales; engañosa si una clase domina.
- **Precisión** $= \frac{TP}{TP+FP}$: de lo que dijo «sube», cuánto subió.
- **Recall** $= \frac{TP}{TP+FN}$: de lo que subió, cuánto detectó.
- **Especificidad** $= \frac{TN}{TN+FP}$: de lo que bajó, cuánto detectó.
- **F1** $= \frac{2PR}{P+R}$: equilibrio entre precisión y recall.
- **Balanced accuracy** $= \frac{\text{recall} + \text{especificidad}}{2}$: promedia las dos clases.
- **MCC** $= \frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$: un número entre −1
  y 1 que no se deja engañar por el desbalance. 1 es perfecto, 0 es azar, −1 es inversión total.
- **Tasa de «sube»**: predicha $= \frac{TP+FP}{N}$ frente a real $= \frac{TP+FN}{N}$; si el modelo
  predice «sube» el 90 % de los días, algo raro pasa.

## 4. Ejemplo a mano

Con predicción $[1, -1, 1, -1, 0]$ y realidad $[1, 1, -1, -1, 0]$:

|  | predicho sube | predicho baja |
| --- | --- | --- |
| **real sube** | 1 (TP) | 1 (FN) |
| **real baja** | 1 (FP) | 1 (TN) |

Un empate (el último 0) queda fuera: cobertura 0.8. Accuracy, precisión, recall, especificidad,
F1 y balanced accuracy valen 0.5; MCC vale 0 (como el azar); las tasas de «sube» son 0.5 y 0.5.

## 5. Implementación

- [`src/sets/direction.ts`](../../src/sets/direction.ts): `confusionMatrix`, `directionReport` y
  `errorsByDirection`.
- [`src/sets/metrics.ts`](../../src/sets/metrics.ts): MAE, RMSE, sesgo, R², correlación y acierto
  direccional.
- [`test/sets/direction.test.ts`](../../test/sets/direction.test.ts): matriz, empates, todo bien, todo
  mal, denominadores cero y error por dirección.
- [`lab/direction/run.ts`](../../lab/direction/run.ts): imprime la matriz y el informe del ejemplo.

## 6. Verificación

- La matriz del ejemplo da TP/FN/FP/TN = 1 y un empate.
- Todo bien lleva accuracy, precisión, recall, F1, balanced y MCC a 1; todo mal lleva MCC a −1.
- Sin positivos predichos o sin positivos reales no hay división entre cero: las métricas indefinidas
  valen 0.
- Un informe sin días direccionales queda a cero y conserva `coverage = 0`.
- 101/101 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Solo binario**: sube/baja; no distingue magnitudes ni varias clases.
- **Excluye los ceros reales**: si hay muchos, la cobertura baja y el informe pierde representatividad.
- **Una predicción 0 es «baja»**: es una convención, no una abstención.
- **No mide magnitud**: un acierto de dirección con un error enorme sigue siendo un acierto; por eso
  se acompaña de MAE/RMSE.
- **Sin calibración ni costes**: no dice si la probabilidad está bien calibrada ni si una estrategia
  gana dinero después de comisiones.
- **`f32`.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Chicco y Jurman (2020), *The advantages of the Matthews correlation coefficient (MCC) over F1 score
  and accuracy*.
- scikit-learn, métricas de clasificación (`confusion_matrix`, `precision_recall_fscore_support`).
