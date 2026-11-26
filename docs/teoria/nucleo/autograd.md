# autograd

## 1. Qué problema resuelve

Entrenar es ajustar los parámetros en la dirección que baja la pérdida, y esa dirección es el
**gradiente**. Derivar a mano cada operación de una red no escala: cambia una capa y hay que rehacer
las cuentas.

El **autograd** construye un grafo mientras se ejecutan las operaciones y calcula todas las derivadas
con la regla de la cadena, en una sola pasada hacia atrás.

## 2. Definición y notación

- **Gradiente** de una salida $L$ respecto de un tensor $x$: el tensor de derivadas parciales
  $\partial L/\partial x$, con la misma forma que $x$.
- **Regla de la cadena**: si $L$ depende de $z$ y $z$ de $x$, entonces
  $\partial L/\partial x = \partial L/\partial z \cdot \partial z/\partial x$.
- **Grafo computacional**: los tensores son nodos y las operaciones son aristas. Cada operación
  guarda cómo repartir el gradiente que recibe entre sus entradas.
- **Semilla**: $\partial L/\partial L = 1$; de ahí arranca el recorrido.
- **Modo reverso**: se recorre el grafo de la salida hacia las entradas, acumulando en cada nodo.

## 3. Derivación

Si los nodos están en orden topológico (cada uno después de sus entradas), el gradiente de un nodo es
la **suma** de las contribuciones de todos los nodos que lo usan. Recorriendo en orden inverso y
sumando, cada arista se procesa una vez.

Para $f = (x + y)\,x$:

$$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial x} = x \cdot 1 + a = 2x + y, \qquad \frac{\partial f}{\partial y} = x.$$

El término $a = x + y$ aparece porque $x$ entra dos veces en el grafo; por eso hay que **acumular**,
no sobrescribir.

## 4. Ejemplo a mano

Con $x = 2$, $y = 3$: $f = (2+3)\cdot 2 = 10$, $\partial f/\partial x = 2\cdot2+3 = 7$ y
$\partial f/\partial y = 2$.

Recorrido hacia atrás:

| Paso | Nodo | Gradiente |
| --- | --- | --- |
| 1 | $f$ | 1 |
| 2 | $a = x+y$ | $\partial f/\partial a = x = 2$ |
| 3 | $x$ (por $a$) | 2 |
| 4 | $x$ (directo) | $a = 5$ |
| 5 | $x$ total | 7 |
| 6 | $y$ | $x = 2$ |

## 5. Implementación

- [`src/core/tensor.ts`](../../src/core/tensor.ts): `requiresGrad`, `grad`, `#parents` y `#backward`.
  `setGraph` engancha el resultado; `addGrad` acumula; `zeroGrad` limpia; `backward` ordena el grafo
  con un recorrido en profundidad, siembra el gradiente de la salida con unos y recorre en orden
  inverso.
- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `add`, `sub` y `mul` registran su grafo
  cuando alguna entrada lo pide. `matmul` usa `dA = G·Bᵀ` y `dB = Aᵀ·G`. `sum` propaga el gradiente
  al eje reducido y `mean` encadena `sum` con el factor `1/cuenta`. `sigmoidTensor`
  ([`src/nn/sigmoid.ts`](../../src/nn/sigmoid.ts)) multiplica por `σ(1−σ)`. `unbroadcast` suma el
  gradiente sobre los ejes expandidos y `broadcastTo` lo expande al reducir; `elementwiseRaw` y
  `unaryRaw` calculan sin registrar, para no ensuciar el grafo durante el backward.
- [`test/core/autograd.test.ts`](../../test/core/autograd.test.ts): gradientes analíticos,
  broadcasting, cadena, `sum`/`mean`, `matmul`, `sigmoid`, `zeroGrad` y comparación con la derivada
  numérica.
- [`lab/autograd/run.ts`](../../lab/autograd/run.ts): deriva `f(x) = (x + x) * x` y comprueba que da
  `4x`.

Todas las operaciones actuales registran gradiente. El test más completo construye un MLP de dos
capas (matmul → add → sigmoid → matmul → add → sigmoid), calcula la pérdida cuadrática media y compara
los gradientes de **todos** los parámetros con diferencias finitas.

Un **MLP** (*multilayer perceptron*, perceptrón multicapa) es una red *feedforward* de capas densas
con una no linealidad entre ellas. Sin la no linealidad, apilar capas equivale a una sola
transformación lineal: `W₂(W₁x) = (W₂W₁)x`.

## 6. Verificación

- `add` reparte el gradiente (uno a cada entrada); `mul` lo multiplica por el valor de la otra.
- Con broadcasting, el gradiente del operando expandido se **suma** en las posiciones repetidas.
- La cadena `(x + x) * x` da `4x`.
- `sum` reparte el gradiente entre los elementos sumados (y `mean` lo divide por la cuenta).
- `matmul` da `dA = G·Bᵀ` y `dB = Aᵀ·G`; `sigmoid` da `σ(1−σ)`.
- La derivada analítica coincide con la numérica (diferencias centrales, tolerancia `1e-2` en `f32`).
- El MLP de dos capas con pérdida cuadrática media pasa el gradcheck en sus cuatro parámetros.
- Sin `requiresGrad` no se construye grafo y no hay gradientes.

Qué fallaría si no se acumulara: en `(x + y) * x`, `x` tendría 2 o 5 en vez de 7.

## 7. Límites y errores comunes

- **`zeroGrad` es manual**: sin limpiar, un segundo `backward` suma al gradiente anterior.
- **Mutar un tensor después del forward** invalida el grafo: los valores se leen en el backward.
- **Sin derivadas de segundo orden** ni *forward-mode*.
- **Solo `f32`**; el gradcheck usa tolerancias amplias por eso.
- **Otras formas de entrenar**: ver [alternativas a backpropagation](../entrenamiento/alternativas.md).

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Karpathy, *micrograd* (autograd escalar mínimo, muy didáctico).
- PyTorch, documentación de *autograd* (grafo dinámico y acumulación).
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 6.5 (retropropagación).
