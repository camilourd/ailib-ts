# Gradiente que se desvanece

## 1. Qué problema resuelve

Una RNN aprende de pasos lejanos solo si el gradiente llega hasta ellos. En una secuencia larga, el
gradiente de la pérdida del último paso respecto de los primeros estados se vuelve tan pequeño que
esos pasos **no aprenden**. Este experimento lo mide.

## 2. Definición y notación

- $L_T$: pérdida en el último paso de una secuencia de longitud $T$.
- $h_t$: estado oculto en el paso $t$.
- $\partial L_T / \partial h_t$: cuánto cambia la pérdida final si cambia el estado $t$.
- **Norma**: $\lVert \partial L_T / \partial h_t \rVert_2$, el tamaño del vector de gradiente.

En BPTT, ese gradiente es un producto de jacobianos desde $t$ hasta $T$; si cada factor tiene norma
menor que 1, el producto se encoge.

## 3. Por qué se desvanece

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b).$$

Al derivar, cada paso aporta un factor con $\tanh' = 1 - \tanh^2 \le 1$ y la matriz $W_h$. Con muchos
pasos, el producto puede caer exponencialmente. Con `f32`, además, los valores pequeños terminan en
**underflow** (cero).

## 4. Experimento

- Modelo entrenado 1500 pasos (Embedding 32, RNN 64).
- Secuencia de **96 pasos**.
- Se calcula la pérdida **solo del último carácter** y se hace backward.
- Se mide la norma del gradiente de cada estado oculto.

| Posición | Norma | log10 |
| --- | --- | --- |
| 0–47 | por debajo de 1e−12 (underflow) | −12 |
| 63 | 8.3e−10 | −9.08 |
| 79 | 2.0e−5 | −4.69 |
| 95 | 0.674 | −0.17 |

El cociente entre la posición 0 y la 95 es **3.65e−26**: el primer estado recibe 26 órdenes de
magnitud menos gradiente que el último. La curva está en
[desvanecimiento.svg](desvanecimiento.svg).

## 5. Implementación

- [`lab/lm/vanishing.ts`](../../lab/lm/vanishing.ts): entrena, elige una ventana de 96, calcula solo la
  pérdida final y dibuja la curva.
- [`src/models/char-lm.ts`](../../src/models/char-lm.ts): `forward` y `lossAndStates` permiten acceder a
  los estados; el laboratorio usa `forward` y la cross-entropy del último paso.
- [`src/nn/rnn.ts`](../../src/nn/rnn.ts): la celda que produce el producto de jacobianos.

## 6. Verificación

- Los estados intermedios reciben gradiente no nulo en el test de `CharLM`.
- En la ventana de 96 pasos, la norma cae de 0.674 en la posición 95 a menos de 1e−12 en las primeras.
- La curva se genera en `docs/teoria/secuencias/desvanecimiento.svg`.
- 123/123 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Una sola corrida y una sola semilla**: el patrón es estructural, pero los números cambian.
- **Solo la pérdida final**: mide el flujo hacia atrás; no es el gradiente de todos los parámetros.
- **Norma, no dirección**: un gradiente pequeño puede ser informativo, pero aquí es ruido/underflow.
- **Sin *gradient clipping*, truncamiento ni GRU/LSTM**: esas soluciones llegan después.
- **`f32`**: parte de la caída es underflow numérico, no solo matemático.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Hochreiter (1991), *Untersuchungen zu dynamischen neuronalen Netzen* (el problema original).
- Bengio, Simard y Frasconi (1994), *Learning long-term dependencies with gradient descent is difficult*.
- Pascanu, Mikolov y Bengio (2013), *On the difficulty of training recurrent neural networks*.
