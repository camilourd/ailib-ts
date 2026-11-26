# Bucle de entrenamiento

## 1. Qué problema resuelve

Ya están las piezas: datos, capa lineal, activación, pérdida, autograd y optimizador. Falta el
**orden** en que se usan. El bucle de entrenamiento es ese ciclo, repetido hasta que la pérdida baja.

## 2. Definición y notación

Un **paso** hace, en este orden:

1. `zeroGrad`: limpia los gradientes del paso anterior.
2. `forward`: calcula la predicción.
3. `loss`: mide el error (un escalar).
4. `backward`: llena los gradientes.
5. `step`: el optimizador actualiza los parámetros.

- **Época**: una pasada completa por el dataset.
- **Lote** (*batch*): el subconjunto de datos que se procesa en un paso.
- **Historial**: la pérdida de cada paso, para ver la curva.

## 3. Por qué este orden

- Si no se limpia el gradiente, el `backward` **suma** al anterior y el paso se hace con una mezcla de
  dos batches.
- El `forward` tiene que ser con los parámetros actuales; el `backward` usa los valores intermedios
  de ese forward.
- El `step` va después del `backward`: primero se calcula la dirección y luego se mueve.
- En esta pieza el lote es **todo el dataset** (full-batch). El ruido del mini-batch y el barajado
  llegan con el pipeline de datos.

## 4. Ejemplo a mano: XOR

XOR no es linealmente separable:

| $x_1$ | $x_2$ | salida |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Ninguna recta separa los unos de los ceros. Con una capa oculta y una no linealidad sí: la red
`2 → 4 (ReLU) → 2` aprende una representación donde el problema es separable; la salida usa
cross-entropy.

Con SGD ($\eta = 0.5$), 3000 pasos y semillas fijas, la pérdida va de 0.797 a 0.0007 y acierta
**4/4**.

## 5. Implementación

- [`src/train/trainer.ts`](../../src/train/trainer.ts): `train(forward, input, options)` ejecuta el
  ciclo, exige una pérdida escalar y devuelve el historial; `onStep` permite mirar la curva.
- [`test/train/trainer.test.ts`](../../test/train/trainer.test.ts): entrena XOR, comprueba el
  historial, `onStep`, cero pasos y que una pérdida no escalar falla.
- [`lab/xor/run.ts`](../../lab/xor/run.ts): imprime la pérdida cada 500 pasos y los aciertos.

## 6. Verificación

- La pérdida de XOR baja de forma monótona en la práctica y termina por debajo de 0.02.
- La red acierta las cuatro filas de XOR.
- `train` con 0 pasos devuelve un historial vacío.
- Una pérdida de rango distinto de 0 falla antes de tocar los parámetros.
- `onStep` recibe el paso y la pérdida de cada iteración.

## 7. Límites y errores comunes

- **Full-batch**: sin lotes, sin barajado y sin épocas explícitas; eso llega con el pipeline de datos.
- **Sin *schedules*, validación, *early stopping* ni checkpoints** todavía.
- **Una sola métrica**: la pérdida; la accuracy se calcula a mano en el laboratorio.
- **Sin reproducibilidad del orden de los datos**: al no barajar, no hay nada que sembrar.
- **Solo `f32` y CPU.**

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Minsky y Papert (1969), *Perceptrons* (el límite lineal y XOR).
- Rumelhart, Hinton y Williams (1986), *Learning representations by back-propagating errors*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 8 (entrenamiento).
