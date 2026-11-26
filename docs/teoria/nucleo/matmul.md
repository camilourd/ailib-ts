# matmul

## 1. Qué problema resuelve

Una capa lineal calcula $y = xW + b$; la atención compara todas las consultas con todas las claves.
Las dos operaciones son productos matriciales: cada salida combina una fila de una matriz con una
columna de la otra. Es la operación que domina el coste de una red.

## 2. Definición y notación

$$(AB)_{ij} = \sum_{k=0}^{K-1} A_{ik} \, B_{kj}$$

Si $A$ es $[M, K]$ y $B$ es $[K, N]$, el resultado es $[M, N]$. El elemento $(i, j)$ es el producto
punto de la **fila** $i$ de $A$ con la **columna** $j$ de $B$.

No conmuta: $AB \neq BA$ en general. La identidad $I$ sí cumple $AI = IA = A$.

## 3. Derivación y algoritmo

La definición ya es el algoritmo: tres bucles (fila, columna, $k$), con coste $O(MNK)$.

El detalle que importa es el **acceso a memoria**: $A$ se recorre por filas y $B$ por columnas. Con
layout row-major, recorrer $B$ por columnas salta de fila en fila; por eso las librerías reales usan
BLAS, bloqueo (*tiling*) y a veces transponen un operando. Aquí usamos los strides, así que funciona
con cualquier tensor, pero sin optimizar el orden de los bucles.

## 4. Ejemplo a mano

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}$$

| | col 0 | col 1 |
| --- | --- | --- |
| fila 0 | $1\cdot7+2\cdot9+3\cdot11 = 58$ | $1\cdot8+2\cdot10+3\cdot12 = 64$ |
| fila 1 | $4\cdot7+5\cdot9+6\cdot11 = 139$ | $4\cdot8+5\cdot10+6\cdot12 = 154$ |

## 5. Implementación

- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `matmul` exige dos matrices de rango 2,
  comprueba que la dimensión interior coincide y recorre con los strides de cada operando. Registra
  gradiente: `dA = G·Bᵀ` y `dB = Aᵀ·G`, con `matmulRaw` para no encadenar grafo durante el backward.
- [`test/core/matmul.test.ts`](../../test/core/matmul.test.ts): el ejemplo de arriba, la identidad,
  un caso $1\times1$ y errores de forma. El gradiente se comprueba en `test/core/autograd.test.ts`.
- [`lab/tensor/ops.ts`](../../lab/tensor/ops.ts): imprime un producto matricial entre los demás
  ejemplos.

## 6. Verificación

- El ejemplo a mano da `[58, 64, 139, 154]`.
- Multiplicar por la identidad devuelve la matriz original.
- Un vector fila por un vector columna da una matriz $1\times1$ con el producto punto.
- Formas incompatibles y tensores que no son de rango 2 fallan.
- El gradcheck contra diferencias finitas coincide para ambos operandos.

Qué fallaría si se intercambiaran los índices de $B$: el resultado sería la transpuesta del correcto
solo en el caso simétrico; en general, valores distintos.

## 7. Límites y errores comunes

- **Solo 2D.** Sin lotes (`[b, m, k] · [b, k, n]`) ni tensores de más rango; llegarán cuando hagan
  falta para la atención.
- **Algoritmo ingenuo** $O(MNK)$. Strassen ($O(n^{2.807})$) y BLAS optimizan, pero no son el objetivo
  de esta pieza.
- **Sin banderas de transposición** (`AᵀB`), aunque `transpose` permite prepararlo copiando.
- **Solo `f32` y CPU.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- BLAS, referencia de las rutinas `gemm` (producto matricial general).
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 2 (álgebra lineal).
- Strassen (1969), el algoritmo subcúbico clásico.
