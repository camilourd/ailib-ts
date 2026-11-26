# Pérdidas

## 1. Qué problema resuelve

El optimizador necesita **un número** que baje: la pérdida. No es lo mismo medir un error de regresión
que la probabilidad que el modelo da a la clase correcta. Aquí hay dos:

- **MSE** para regresión.
- **Entropía cruzada** para clasificación.

## 2. Definición y notación

**MSE**:

$$\mathrm{MSE}(\hat y, y) = \frac{1}{N}\sum_i (\hat y_i - y_i)^2$$

Misma forma en predicción y objetivo; $N$ es el número total de elementos.

**Entropía cruzada** con logits $z$ de forma $[\text{lote}, \text{clases}]$ y la clase correcta $y$
por fila:

$$p_i = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad L = -\ln p_y = \ln\!\left(\sum_j e^{z_j}\right) - z_y.$$

Los **logits** son las salidas sin normalizar de la última capa; **softmax** las convierte en
probabilidades. El objetivo se pasa como **índice** de clase (`0..C-1`), no como one-hot.

Las dos devuelven un escalar: la **media** del lote.

## 3. Derivación

**MSE**: $\partial L/\partial \hat y_i = 2(\hat y_i - y_i)/N$. Sale de componer `sub`, `mul` y `mean`;
el autograd lo hace solo.

**Entropía cruzada**: el gradiente respecto de los logits es sorprendentemente simple:

$$\frac{\partial L}{\partial z_i} = p_i - \mathbb{1}[i = y].$$

Es softmax menos el one-hot de la clase correcta. Con la media del lote, se divide por el tamaño del
lote. No hay que derivar el logaritmo ni el máximo a mano: el backward escribe esa resta.

**Estabilidad**: `exp(1000)` es `Infinity`. Como softmax es invariante a un desplazamiento, se resta el
máximo por fila antes de exponenciar:

$$m = \max_j z_j, \qquad L = m + \ln\!\left(\sum_j e^{z_j - m}\right) - z_y.$$

Los exponentes quedan $\le 0$ y no desbordan.

## 4. Ejemplo a mano

**MSE** con $\hat y = [1, 2]$ e $y = [0, 0]$: diferencias $[1, 2]$, cuadrados $[1, 4]$, media $2.5$;
gradiente $2\cdot[1,2]/2 = [1,2]$.

**Entropía cruzada** con $z = [1, 2, 3]$ y clase correcta 2:

- $m = 3$; $e^{z-m} = [0.1353, 0.3679, 1]$; suma $1.5032$; $\ln = 0.4076$.
- $L = 3 + 0.4076 - 3 = 0.4076$ (que es $-\ln 0.6652$).
- Gradiente: softmax $[0.0900, 0.2447, 0.6652]$ menos one-hot en 2 → $[0.0900, 0.2447, -0.3348]$.

## 5. Implementación

- [`src/nn/losses.ts`](../../src/nn/losses.ts): `mse` se compone con `sub`/`mul`/`mean`.
  `crossEntropy` va **fundida**: forward estable con el máximo por fila y backward propio con
  `softmax − onehot`, dividido por el lote. No necesita `softmax` ni `gather` como operaciones sueltas.
- [`test/nn/losses.test.ts`](../../test/nn/losses.test.ts): ejemplo a mano, media del lote,
  estabilidad con logits de ±1000, gradcheck numérico y errores de forma/objetivo.
- [`lab/losses/run.ts`](../../lab/losses/run.ts): un clasificador lineal de 3 clases con SGD y
  cross-entropy.

## 6. Verificación

- MSE da 2.5 y sus gradientes `[1, 2]` y `[-1, -2]`.
- Cross-entropy da 0.4076 y el gradiente de softmax menos one-hot.
- La media del lote y su gradiente coinciden con el cálculo a mano.
- Con logits de 1000 y 1001 no hay `Infinity` ni `NaN`.
- Ambas pasan el gradcheck contra diferencias finitas.
- Objetivos fuera de rango, no enteros, lotes vacíos o formas incompatibles fallan.

## 7. Límites y errores comunes

- **Solo reducción media**; sin `sum` ni `none`.
- **Cross-entropy multiclase por índices**: sin label smoothing, pesos por clase ni BCE binaria.
- **Logits de rango 2**; sin tensores de más dimensiones.
- **`log(0)` y `log` de negativos** dan `-Infinity`/`NaN`; el softmax estable evita el desbordamiento,
  pero un logit `-Infinity` sigue siendo un caso límite.
- **Solo `f32`.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 4 y 6 (pérdidas y unidades de salida).
- PyTorch, documentación de `torch.nn.CrossEntropyLoss` (fusiona softmax y NLL).
- El truco *log-sum-exp* para estabilidad numérica.
