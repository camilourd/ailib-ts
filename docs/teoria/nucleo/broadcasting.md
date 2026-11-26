# broadcasting

## 1. Qué problema resuelve

Sumar un sesgo por neurona a un lote de activaciones es lo más común del mundo: el lote tiene forma
`[2, 3]` y el sesgo `[3]`. Sin una regla, habría que copiar el sesgo dos veces o escribir un bucle
distinto por cada forma.

El **broadcasting** define cómo combinar tensores de formas distintas sin copiar los operandos: las
dimensiones que no encajan se tratan como si tuvieran tamaño 1 y se recorren con stride 0.

## 2. Definición y reglas

Se alinean las formas **por la derecha** y se comparan dimensión a dimensión:

- si son iguales, el resultado toma ese tamaño;
- si una de las dos es 1, el resultado toma el tamaño de la otra;
- si falta una dimensión, se trata como 1;
- en cualquier otro caso, las formas son incompatibles y hay error.

Ejemplos:

- `[2, 3]` con `[3]` → `[2, 3]`
- `[2, 1]` con `[1, 3]` → `[2, 3]`
- `[]` (escalar) con `[2, 3]` → `[2, 3]`
- `[2, 3]` con `[2]` → error: 3 contra 2.

La expansión es **virtual**: en el eje que se repite, el stride del operando es 0, así que todos los
índices leen el mismo valor. El resultado sí es un tensor nuevo; lo que no se copia es el operando.

## 3. Por qué funciona

Un elemento del resultado se localiza con los strides del resultado. Para cada eje, el índice se
multiplica por el stride del operando; si ese stride es 0, el valor no cambia al avanzar en ese eje.
No hace falta materializar la copia expandida: basta con calcular el offset.

Esto es lo mismo que hace NumPy y lo que permite escribir `sesgo + lote` sin pensar en el tamaño del
lote.

## 4. Ejemplo a mano

`[[1, 2, 3], [4, 5, 6]] + [10, 20, 30]`:

| | col 0 | col 1 | col 2 |
| --- | --- | --- | --- |
| fila 0 | 11 | 22 | 33 |
| fila 1 | 14 | 25 | 36 |

`[[1], [2]] * [[10, 20, 30]]` (formas `[2, 1]` y `[1, 3]` → `[2, 3]`):

| | col 0 | col 1 | col 2 |
| --- | --- | --- | --- |
| fila 0 | 10 | 20 | 30 |
| fila 1 | 20 | 40 | 60 |

## 5. Implementación

- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `broadcastShapes` aplica las reglas y
  `broadcastStrides` devuelve los strides vistos desde la forma resultante, con 0 en los ejes
  expandidos. `add`, `sub` y `mul` son elementales: calculan el offset de cada operando y combinan.
  Si alguna entrada pide gradiente, lo registran y `unbroadcast` lo suma en los ejes expandidos.
- [`test/core/tensor-ops.test.ts`](../../test/core/tensor-ops.test.ts): reglas, strides 0, valores y
  errores de formas incompatibles.
- [`lab/tensor/ops.ts`](../../lab/tensor/ops.ts): imprime un lote, un sesgo, su suma y el producto de
  una columna por una fila.

## 6. Verificación

- `broadcastShapes([2, 3], [3])` da `[2, 3]`; `[2, 1]` con `[1, 3]` da `[2, 3]`; `[2, 3]` con `[2]`
  falla.
- `broadcastStrides([3], [1], [2, 3])` da `[0, 1]`; un escalar da `[0, 0]`.
- La suma y el producto coinciden con el cálculo a mano.
- Un escalar se combina con cualquier forma.

Qué fallaría si el stride broadcast no fuera 0: se leería un elemento que no toca (o fuera de rango) y
la suma daría valores equivocados en las filas a partir de la segunda.

## 7. Límites y errores comunes

- **El resultado se materializa**: no hay vistas; el operando no se copia, el resultado sí.
- **Sin operaciones *in-place***: `add` devuelve un tensor nuevo.
- **`matmul` no hace broadcasting**: exige formas exactas (llega en su propia pieza).
- **Sin `keepdims`** ni ejes negativos.
- **Solo `f32`**: no hay promoción de tipos.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- NumPy, *Broadcasting* (reglas y ejemplos).
- PyTorch, documentación de *broadcasting semantics*.
