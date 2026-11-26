# Atención

## 1. Qué problema resuelve

Una RNN comprime el pasado en un estado; la atención deja que cada posición **mire directamente** a las
demás y decida a cuáles darles peso. Es la pieza que hace posible el transformer.

## 2. Definición y notación

Para una secuencia de $L$ posiciones y dimensión $d$:

- $Q$: consultas (*queries*), $[L, d]$.
- $K$: claves (*keys*), $[L, d]$.
- $V$: valores (*values*), $[L, d_v]$.

$$S = \frac{QK^{T}}{\sqrt{d}}, \qquad P = \mathrm{softmax}(S), \qquad O = PV.$$

- $S$: puntuaciones de afinidad, $[L, L]$.
- $P$: pesos de atención, una distribución por fila.
- $O$: salida, $[L, d_v]$.

Con **máscara causal**, $S_{ij} = -\infty$ para $j > i$: una posición solo puede mirar hacia atrás.
El factor $1/\sqrt{d}$ evita que los productos crezcan con la dimensión.

## 3. Por qué funciona

- La consulta de una posición se compara con todas las claves; el softmax convierte esa comparación en
  pesos que suman 1.
- La salida es una mezcla ponderada de los valores: información de otras posiciones, sin recurrencia.
- La máscara causal convierte la atención en un modelo autorregresivo: el token $t$ no ve $t+1$.

## 4. Ejemplo a mano

$Q = [1, 1]^{T}$, $K = [1, 0]^{T}$, $V = [2, 4]^{T}$, sin máscara y $d=1$:

- $S = [1, 0]$; $P = [0.731, 0.269]$.
- $O = 2\cdot0.731 + 4\cdot0.269 = 2.538$.

Con causal y tres posiciones iguales, la primera fila solo mira a la primera; la segunda reparte
entre las dos primeras; la tercera, entre las tres. La matriz queda triangular inferior.

## 5. Implementación

- [`src/nn/attention.ts`](../../src/nn/attention.ts): `scaledDotProductAttention(Q, K, V, { causal })`
  con forward y backward propios; devuelve `output` y `weights`.
- [`test/nn/attention.test.ts`](../../test/nn/attention.test.ts): ejemplo a mano, máscara causal,
  validación de formas y gradcheck de Q/K/V.
- [`lab/attention/run.ts`](../../lab/attention/run.ts): imprime los pesos y la salida de cuatro
  posiciones.

El backward usa:

$$dV = P^{T}\,dO, \qquad dP = dO\,V^{T}, \qquad dS = P \odot (dP - \mathrm{rowsum}(P \odot dP)),$$

y de ahí $dQ = dS\,K/\sqrt{d}$ y $dK = dS^{T}Q/\sqrt{d}$.

## 6. Verificación

- El ejemplo a mano da los pesos 0.731/0.269 y la salida 2.538.
- La máscara causal deja ceros por encima de la diagonal y las salidas 1, 1.5 y 2.
- Formas incompatibles fallan.
- El gradiente de Q, K y V coincide con diferencias finitas.
- 138/138 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Una sola secuencia**: sin batch.
- **Una sola cabeza**: multi-head llega después.
- **Sin dropout** ni proyecciones de salida.
- **Los pesos se devuelven sin gradiente**: son para mirar, no para entrenar.
- **Máscara causal fija**: no hay máscara de padding todavía.
- **Costo $O(L^2)$**: por eso reducir tokens con BPE importa.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Vaswani et al. (2017), *Attention Is All You Need*.
- Bahdanau, Cho y Bengio (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Documentación de `torch.nn.functional.scaled_dot_product_attention`.
