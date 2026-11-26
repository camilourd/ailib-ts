# ReLU

## 1. Qué problema resuelve

Una red necesita una **no linealidad** entre capas; sin ella, apilar capas lineales equivale a una
sola. La ReLU (*rectified linear unit*) es la más usada: barata y sin saturación en el lado positivo.

## 2. Definición y notación

$$\mathrm{ReLU}(x) = \max(0, x)$$

- Para $x < 0$: 0.
- Para $x > 0$: $x$.
- Su derivada es 0 si $x < 0$ y 1 si $x > 0$; en $x = 0$ no es derivable (se elige 0).

## 3. Por qué funciona

- **Barata**: una comparación, sin exponenciales.
- **No satura en positivo**: el gradiente vale 1, así que no se apaga como la sigmoid.
- **Dispersa**: muchas salidas quedan a 0, lo que ayuda a la representación.

### Por qué el lado negativo es un problema

No es que los valores negativos sean malos: es que su derivada es **0**, así que no hay señal para
corregirlos. Si una neurona cae siempre en la zona negativa —sesgo muy negativo, tasa de aprendizaje
alta o mala inicialización—, su gradiente es 0 para todos los ejemplos y sus pesos dejan de
actualizarse: la neurona está **muerta**. Con SGD simple no se recupera y el ancho efectivo de la red
se encoge.

La mitad negativa también puede llevar información; ReLU la descarta. LeakyReLU la conserva escalada.

### La familia

| Activación | Lado negativo | Derivada negativa | Nota |
| --- | --- | --- | --- |
| ReLU | 0 | 0 | la más simple y rápida |
| LeakyReLU | $\alpha x$ ($\alpha \approx 0.01$) | $\alpha$ | deja una rendija para recuperarse |
| PReLU | $\alpha x$, $\alpha$ aprendido | $\alpha$ | $\alpha$ se entrena |
| ELU | $\alpha(e^x - 1)$ | $\alpha e^x$ | suave y centrada cerca de 0 |
| GELU / SiLU | suave | $>0$ | habituales en transformers |

ReLU ganó en CNN y MLP por coste y gradiente 1; LeakyReLU es el parche habitual contra las neuronas
muertas; en transformers GELU y SiLU son la norma.

## 4. Ejemplo a mano

Para $x = [-2, 0, 3]$:

| $x$ | ReLU | derivada |
| --- | --- | --- |
| -2 | 0 | 0 |
| 0 | 0 | 0 |
| 3 | 3 | 1 |

## 5. Implementación

- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `relu(tensor)` aplica `max(0, x)` y
  registra el gradiente 0/1 cuando el tensor lo pide.
- [`test/core/unary-ops.test.ts`](../../test/core/unary-ops.test.ts): valores y gradiente.
- [`lab/linear/run.ts`](../../lab/linear/run.ts) usa la capa lineal; la ReLU se probará en el MLP.

## 6. Verificación

- `relu([-2, 0, 3])` da `[0, 0, 3]`.
- Su gradiente es `[0, 0, 1]`; los negativos no reciben gradiente.
- Se encadena con `sum`/`matmul` sin tocar el autograd.

## 7. Límites y errores comunes

- **Dying ReLU**: si una neurona se queda siempre negativa, su gradiente es 0 para siempre. LeakyReLU
  o inicializaciones cuidadas lo mitigan.
- **No derivable en 0**: se define la derivada como 0; es una convención.
- **Sin variantes** todavía (LeakyReLU, GELU, SiLU).

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Nair y Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 6 (unidades ocultas).
