# MLP

## 1. Qué problema resuelve

Una capa lineal sola no puede resolver XOR ni casi ninguna función no lineal: es una transformación
afín. El **MLP** (*multilayer perceptron*, perceptrón multicapa) apila capas lineales con una no
linealidad entre ellas para ganar capacidad.

## 2. Definición y notación

Un MLP con tamaños $[n_0, n_1, \dots, n_L]$ aplica:

$$h_0 = x, \qquad h_l = \mathrm{ReLU}(h_{l-1}W_l + b_l) \text{ para } l < L, \qquad y = h_{L-1}W_L + b_L.$$

- Cada $W_l$ es $[n_{l-1}, n_l]$; cada $b_l$ es $[n_l]$.
- Las capas ocultas usan ReLU; la última es lineal (logits o regresión).
- Sin la ReLU, apilar capas equivale a una sola transformación lineal.

## 3. Por qué funciona

Cada capa aprende una representación nueva; la no linealidad permite que la composición no colapse.
Con una capa oculta suficiente, un MLP puede aproximar funciones continuas (teorema de aproximación
universal), aunque eso no dice cuántos datos ni cuánto entrenamiento hacen falta.

## 4. Ejemplo a mano

XOR con `2 → 4 → 2`: la capa oculta transforma las cuatro entradas en una representación donde las
clases ya son separables; la salida calcula dos logits y la cross-entropy los convierte en pérdida.
Con SGD y semillas fijas, el modelo pasa de 0.797 a 0.0007 y acierta 4/4.

## 5. Implementación

- [`src/nn/mlp.ts`](../../src/nn/mlp.ts): `MLP(sizes, { seed })` construye una `Linear` por tramo;
  `forward` aplica ReLU entre capas y deja la última lineal; `parameters()` aplana las de todas.
- [`src/nn/model.ts`](../../src/nn/model.ts): el contrato `Model` (`forward` + `parameters`) que
  comparten `Linear`, `MLP` y los futuros modelos.
- [`test/nn/mlp.test.ts`](../../test/nn/mlp.test.ts): forma de salida, parámetros y gradientes.

## 6. Verificación

- `MLP([3, 4, 1])` recibe `[n, 3]` y devuelve `[n, 1]`; tiene 4 tensores de parámetros (dos por capa).
- El gradiente llega a todos los parámetros y a la entrada.
- Tamaños no positivos o no enteros fallan.
- El MLP de XOR converge (ver `lab/xor/run.ts`).

## 7. Límites y errores comunes

- **Solo denso y 2D**: sin convoluciones, recurrencia ni atención.
- **ReLU fija**: sin LeakyReLU, GELU ni otras; se añadirán cuando se necesiten.
- **Sin *dropout*, normalización ni regularización**.
- **Sin inicialización configurable**: cada capa usa Xavier con `seed + índice`.
- **Más capas no siempre es mejor**: sin datos y sin cuidado, sobreajusta.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 6 (redes *feedforward*).
- Cybenko (1989) y Hornik et al. (1989), teoremas de aproximación universal.
