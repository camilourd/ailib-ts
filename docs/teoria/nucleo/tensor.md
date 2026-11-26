# tensor

## 1. Qué problema resuelve

Una red procesa lotes de vectores, matrices y volúmenes: pesos, activaciones, imágenes de
alto×ancho×canales. Con arrays de arrays, cada operación recorre estructuras irregulares y el código
se llena de bucles anidados que hay que validar una y otra vez.

Un **tensor** empaqueta los números en un solo buffer y guarda aparte la **forma**. Así una operación
recorre memoria contigua, la forma se valida una vez y el mismo tipo sirve para un vector, una matriz
o un lote.

### Por qué un tensor y qué alternativas hay

- **Ventajas**: memoria contigua y homogénea (vectorización, GPU, BLAS), forma explícita (operaciones
  genéricas y batching), strides y vistas (reshape y transpose baratos), una sola abstracción para
  vector, matriz, lote e imagen, y base natural para autograd y compiladores.
- **Alternativas**: arrays anidados (flexibles pero irregulares, lentos y sin vectorización),
  representaciones dispersas CSR/COO (cuando casi todo es cero), tensores *ragged* (secuencias de
  longitud variable), dataframes/Arrow (tabular, no cálculo denso) y layouts de hardware
  (NCHW/NHWC, *tiled*). Los dos últimos siguen siendo tensores, solo cambia el orden o el dtype.

## 2. Definición y notación

Un tensor es un arreglo $n$-dimensional de números:

- **datos**: el buffer plano, en este proyecto un `Float32Array`.
- **forma** (`shape`): el tamaño de cada dimensión.
- **rango** (`rank`): cuántas dimensiones tiene.
- **tamaño** (`size`): el producto de la forma, cuántos números hay.

Escalar: forma `[]` (rango 0, tamaño 1). Vector: `[n]`. Matriz: `[m, n]`. Lote de matrices: `[b, m, n]`.

## 3. Layout y strides

Los datos viven en una sola fila. El **layout row-major** dice que el último índice varía más rápido:
para forma `[2, 3]` los datos van `[a00, a01, a02, a10, a11, a12]`.

Los **strides** dicen cuántos elementos hay que saltar en memoria para avanzar uno en cada dimensión.
Con forma `[2, 3]` son `[3, 1]`: avanzar en la primera dimensión salta 3, en la segunda salta 1.

Se calculan de atrás hacia delante, con el último igual a 1:

$$s_{n-1} = 1, \qquad s_k = s_{k+1} \cdot a_{k+1}$$

donde $a_k$ es el tamaño de la dimensión $k$. Para `[2, 3, 4]`: `[12, 4, 1]`.

Con eso, la dirección de un elemento es una suma:

$$\text{offset}(i_0, \dots, i_{n-1}) = \sum_k i_k \, s_k$$

Por eso un tensor es **datos + forma**: la forma decide qué índices son válidos y los strides deciden
dónde está cada uno. Cambiar la forma sin mover los datos es lo que hace `reshape`; cambiar el orden de
los ejes sin moverlos es lo que hace una vista de `transpose`. En esta pieza ambas cosas copian; las
vistas de verdad necesitan además un `offset` y strides no canónicos (ver §7).

## 4. Ejemplo a mano

Datos `[1, 2, 3, 4, 5, 6]` con forma `[2, 3]`:

| | col 0 | col 1 | col 2 |
| --- | --- | --- | --- |
| fila 0 | 1 | 2 | 3 |
| fila 1 | 4 | 5 | 6 |

Con strides `[3, 1]`, el elemento `(1, 2)` está en el offset $1 \cdot 3 + 2 \cdot 1 = 5$, que es el
valor 6.

`transpose()` da forma `[3, 2]` y datos `[1, 4, 2, 5, 3, 6]`:

| | col 0 | col 1 |
| --- | --- | --- |
| fila 0 | 1 | 4 |
| fila 1 | 2 | 5 |
| fila 2 | 3 | 6 |

El elemento `(2, 0)` de la transpuesta está en el offset $2 \cdot 2 + 0 = 4$, que es el valor 3.

## 5. Implementación

- [`src/core/tensor.ts`](../../src/core/tensor.ts): la clase `Tensor` con `data`, `shape`, `strides` y
  `size`. Constructores `zeros`, `ones` y `from`; acceso con `get`/`set`; `reshape` y `transpose`
  devuelven tensores nuevos (copian).
- [`test/core/tensor.test.ts`](../../test/core/tensor.test.ts): strides, acceso, errores de índices,
  `reshape`, `transpose` y un caso de rango 3.
- [`lab/tensor/run.ts`](../../lab/tensor/run.ts): imprime un tensor, su transpuesta y su `reshape`.

## 6. Verificación

- Strides de `[2, 3]` son `[3, 1]` y los de `[2, 3, 4]` son `[12, 4, 1]`.
- `get` y `set` recorren la misma dirección que los strides.
- `reshape` conserva el orden row-major y no comparte datos con el original.
- `transpose` coincide con el cálculo a mano, también con un orden de ejes explícito.
- Índices fuera de rango, número de índices equivocado, datos de longitud distinta a la forma y
  órdenes de ejes inválidos fallan con un error claro.

Qué fallaría si los strides estuvieran mal: `get(1, 2)` devolvería otro elemento. Si `transpose`
mapeara mal los ejes, la matriz transpuesta tendría los valores correctos solo en la diagonal.

## 7. Límites y errores comunes

- **Sin vistas todavía.** `reshape` y `transpose` copian. Una vista real necesita `offset` y strides
  no canónicos; se añadirá cuando el rendimiento importe (F7).
- **Sin slicing** (`t[1:3]`) ni indexado booleano.
- **Sin broadcasting**: sumar formas distintas llega en la pieza de operaciones.
- **Solo `f32` y CPU.** `0.1` no se representa exacto en `Float32Array`; los tests usan valores
  representables.
- **La forma no se valida en cada operación todavía**; cada función nueva debe hacerlo.
- **Investigación abierta.** El tensor denso y contiguo no está en discusión; lo que se investiga es
  cómo almacenarlo y moverlo: layouts automáticos y compiladores (XLA, TVM, Triton, MLIR), formatos
  numéricos en bloque (MXFP4/8), dispersión estructurada (2:4), tensores *ragged* y formas dinámicas,
  y sharding distribuido (GSPMD, DTensor).

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- NumPy, *Internal memory layout of an ndarray* (strides y contigüidad).
- PyTorch, documentación de `Tensor.view` y `Tensor.stride`.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 2 (álgebra lineal y tensores).
- TACO, *Tensor Algebra Compiler* (representaciones dispersas y sus layouts).
- GSPMD y DTensor (particionado de tensores en computación distribuida).
