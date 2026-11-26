# reducciones

## 1. Qué problema resuelve

La pérdida de un lote suele ser la **media** de los errores de cada ejemplo; normalizar una matriz
por columnas exige **sumar** por columnas; y muchas operaciones necesitan colapsar un eje. Reducir es
pasar de un tensor a otro con menos dimensiones sumando (o promediando) a lo largo de un eje.

## 2. Definición y notación

- `sum(t)` sin eje: suma todos los elementos y devuelve un escalar (rango 0).
- `sum(t, axis)`: suma a lo largo de ese eje; **el eje desaparece** del resultado.
- `mean` es `sum` dividido por cuántos elementos se sumaron: el tamaño del eje, o el tamaño total si
  no hay eje.

Si $t$ tiene forma $[2, 3]$:

- `sum(t, 0)` tiene forma $[3]$ (una suma por columna);
- `sum(t, 1)` tiene forma $[2]$ (una suma por fila);
- `sum(t)` es un escalar.

## 3. Por qué desaparece el eje

El resultado responde «cuánto suma esta columna» (o esta fila): ya no hay una dimensión que recorrer,
hay un valor por cada posición de las demás. Por eso el rango baja en uno.

El coste es $O(\text{size})$: se recorre cada elemento una vez y se acumula en la posición que le toca
del resultado.

## 4. Ejemplo a mano

Para $t = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$:

| Operación | Resultado |
| --- | --- |
| `sum(t)` | 10 |
| `sum(t, 0)` | `[4, 6]` |
| `sum(t, 1)` | `[3, 7]` |
| `mean(t)` | 2.5 |
| `mean(t, 0)` | `[2, 3]` |
| `mean(t, 1)` | `[1.5, 3.5]` |

## 5. Implementación

- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `sum` descompone el índice lineal del
  tensor en índices por eje, omite el eje reducido y acumula en el offset del resultado. Si el tensor
  pide gradiente, lo expande al eje reducido (`broadcastTo`). `mean` reusa `sum` y multiplica por el
  escalar $1/\text{cuenta}$ con broadcasting.
- [`test/core/reductions.test.ts`](../../test/core/reductions.test.ts): total, por eje, escalar,
  errores y un eje de tamaño 0.
- [`lab/tensor/ops.ts`](../../lab/tensor/ops.ts): imprime una suma y una media entre los demás
  ejemplos.

## 6. Verificación

- `sum` y `mean` coinciden con la tabla de arriba.
- Un escalar se reduce a sí mismo.
- Un eje fuera de rango o no entero falla.
- Un eje de tamaño 0 da ceros (si quedan posiciones) o un resultado vacío.
- El gradiente de `sum` es 1 por elemento; el de `mean`, $1/\text{cuenta}$; se comprueba en
  `test/core/autograd.test.ts`.

Qué fallaría con un eje mal calculado: `sum(t, 0)` devolvería la forma de `sum(t, 1)` o valores
mezclados entre filas y columnas.

## 7. Límites y errores comunes

- **Sin `keepDims`**: el eje siempre desaparece.
- **Un solo eje por llamada**; reducir dos a la vez exige dos pasos.
- **Sin ejes negativos** (`-1` no significa «el último»).
- **Media de un eje vacío**: la cuenta es 0 y el resultado es `NaN`; no se ha definido un valor por
  defecto.
- **Precisión `f32`**: sumar muchos valores pequeños acumula error; para sumas largas se usan
  compensaciones (Kahan) o `f64`.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- NumPy, `ndarray.sum` y `ndarray.mean` (semántica de `axis` y `keepdims`).
- Kahan (1965), suma compensada para reducir el error de redondeo.
