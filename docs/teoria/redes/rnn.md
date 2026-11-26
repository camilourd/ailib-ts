# RNN

## 1. Qué problema resuelve

Un MLP ve una entrada y olvida. Una secuencia necesita **estado**: lo que ya se leyó tiene que influir
en lo que viene. La RNN de Elman mantiene un vector oculto que se actualiza en cada paso.

## 2. Definición y notación

$$h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$$

- $x_t$: entrada del paso $t$, forma $[1, \text{entrada}]$.
- $h_{t-1}$: estado anterior, forma $[1, \text{oculto}]$.
- $W_x$: $[\text{entrada}, \text{oculto}]$; $W_h$: $[\text{oculto}, \text{oculto}]$; $b$:
  $[\text{oculto}]$.
- $h_t$: estado nuevo, $[1, \text{oculto}]$.

El estado inicial es un vector de ceros. La misma celda se reutiliza en todos los pasos: los pesos se
comparten.

## 3. Por qué funciona y por qué duele

- **Memoria**: $h_t$ resume el pasado; el modelo puede recordar «lo que venía antes».
- **BPTT**: el backward recorre la secuencia al revés, aplicando la regla de la cadena en cada paso.
  No hay código manual: el grafo define-by-run ya conecta los pasos.
- **El problema**: para llegar a un paso lejano hay que multiplicar muchos jacobianos. Con $\tanh' \le 1$, ese producto se encoge y el gradiente **se desvanece**; por eso una RNN simple no recuerda bien secuencias largas.

## 4. Ejemplo a mano

Con $W_x = 1$, $W_h = 1$, $b = 0$, $x = 0.5$ y $h_0 = 0$:

- $h_1 = \tanh(0.5) = 0.4621$.
- $h_2 = \tanh(0.5 + 0.4621) = \tanh(0.9621) = 0.7450$.

El segundo estado ya no es solo la entrada: incorpora el primero. Eso es la memoria.

## 5. Implementación

- [`src/nn/rnn.ts`](../../src/nn/rnn.ts): `RNN(entrada, oculto, { seed })` con `step`, `forward`,
  `initialHidden` y `parameters`. Xavier uniforme en las dos matrices; sesgo a cero.
- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `tanh` con su derivada
  $1 - \tanh^2$.
- [`test/nn/rnn.test.ts`](../../test/nn/rnn.test.ts): formas, secuencia, gradiente, determinismo y
  validaciones.
- [`lab/rnn/run.ts`](../../lab/rnn/run.ts): cinco pasos y normas de gradiente.

## 6. Verificación

- `initialHidden()` da `[1, oculto]` en ceros y sin gradiente.
- `step` y `forward` conservan `[1, oculto]` en cada paso.
- El gradiente llega a los tres tensores de parámetros y a la entrada.
- Dos RNN con la misma semilla tienen los mismos pesos.
- Formas inválidas y tamaños no positivos fallan.
- 119/119 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Un solo ejemplo por paso** (`[1, ...]`): sin batch.
- **Sin truncamiento de BPTT**: secuencias largas acumulan grafo y gradiente que se desvanece.
- **Sin GRU/LSTM**: las compuertas llegan después.
- **Sin bidireccionalidad ni máscaras**.
- **El estado inicial siempre es cero**: no se puede continuar un estado guardado todavía.
- **`tanh` satura**: en valores grandes la derivada tiende a 0.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Elman (1990), *Finding Structure in Time*.
- Rumelhart, Hinton y Williams (1986), *Learning representations by back-propagating errors*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 10 (secuencias).
