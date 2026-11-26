# Capa lineal

## 1. Qué problema resuelve

Una red necesita transformar un vector de características en otro: pasar de 2 entradas a 3 salidas,
proyectar una representación, reducir dimensión. La **capa lineal** (también *densa* o *fully
connected*) hace esa transformación afín: cada salida es una combinación lineal de la entrada más un
sesgo.

## 2. Definición y notación

$$y = xW + b$$

- $x$: entrada $[\text{lote}, \text{entrada}]$.
- $W$: pesos $[\text{entrada}, \text{salida}]$.
- $b$: sesgo $[\text{salida}]$.
- $y$: salida $[\text{lote}, \text{salida}]$.

Cada **columna** de $W$ es una neurona: su producto punto con la entrada más su sesgo. El lote se
procesa de una vez porque `matmul` ya lo permite.

## 3. Derivación (lo que hace el autograd)

Con $G = \partial L/\partial y$ de forma $[\text{lote}, \text{salida}]$:

$$\frac{\partial L}{\partial W} = x^{T}G, \qquad \frac{\partial L}{\partial b} = \sum_{\text{lote}} G, \qquad \frac{\partial L}{\partial x} = G W^{T}.$$

No hay código nuevo: sale de encadenar `matmul` (que ya sabe su backward) con `add` (que reparte el
gradiente del sesgo con broadcasting y lo suma en el lote).

## 4. Ejemplo a mano

Con $x = [1, 2]$, $W = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}$ y $b = [10, 20]$:

$$xW = [1\cdot1 + 2\cdot3,\; 1\cdot2 + 2\cdot4] = [7, 10], \qquad y = [17, 30].$$

## 5. Implementación

- [`src/nn/linear.ts`](../../src/nn/linear.ts): `Linear` con `W`, `b` y `forward(x)`. La inicialización
  es Xavier uniforme, $U(-\sqrt{6/(\text{entrada}+\text{salida})}, +\sqrt{6/(\dots)})$, con `seed` o
  `rng` para que sea reproducible; el sesgo empieza en 0. `parameters()` devuelve ambos para el
  optimizador.
- [`test/nn/linear.test.ts`](../../test/nn/linear.test.ts): ejemplo a mano, reproducibilidad y rango
  de la inicialización, llegada del gradiente, formas y errores.
- [`lab/linear/run.ts`](../../lab/linear/run.ts): una capa 2 → 3 sobre un lote de dos vectores.

## 6. Verificación

- El ejemplo a mano da `[17, 30]`.
- Dos capas con la misma semilla tienen los mismos pesos; todos caen dentro del límite de Xavier.
- `weights.grad`, `bias.grad` y `input.grad` tienen la forma correcta tras `backward`.
- Entradas que no son `[lote, entrada]` y tamaños no positivos fallan.

Qué fallaría si el sesgo no se emitiera: la capa sería lineal pura y no podría desplazar la salida.

## 7. Límites y errores comunes

- **Solo 2D** (`[lote, entrada]`); sin tensores de más rango todavía.
- **Siempre con sesgo**; no hay opción de desactivarlo.
- **Solo Xavier uniforme**; He, normal y otras inicializaciones llegarán después.
- **Sin activación dentro**: la capa es afín a propósito; la no linealidad se añade aparte.
- **Solo `f32` y CPU.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 6 (capas y unidades ocultas).
- Glorot y Bengio (2010), *Understanding the difficulty of training deep feedforward networks*.
- PyTorch, documentación de `torch.nn.Linear`.
