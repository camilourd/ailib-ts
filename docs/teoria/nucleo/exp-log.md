# exp y log

## 1. Qué problema resuelve

La exponencial y el logaritmo aparecen en todas las probabilidades: softmax, entropía cruzada,
verosimilitud, estabilidad numérica. Son inversas y sus derivadas son sencillas, lo que las hace
buenas para el autograd.

## 2. Definición y notación

- $\exp(x) = e^x$, definida para todo $x$, siempre positiva.
- $\ln(x)$, definida para $x > 0$; $\ln(1) = 0$, $\ln(e) = 1$.
- Son inversas: $\ln(\exp(x)) = x$ y $\exp(\ln(x)) = x$ (para $x > 0$).

## 3. Derivación

$$\frac{d}{dx}e^x = e^x, \qquad \frac{d}{dx}\ln x = \frac{1}{x}.$$

Con la regla de la cadena, para una composición $f(g(x))$:

$$\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x).$$

Por eso el gradiente de `log` es $1/x$ y el de `exp` es el propio valor de salida.

## 4. Ejemplo a mano

| $x$ | $e^x$ | $\ln x$ |
| --- | --- | --- |
| 0 | 1 | — |
| 1 | 2.7183 | 0 |
| $e$ | — | 1 |
| 2 | 7.3891 | 0.6931 |

## 5. Implementación

- [`src/core/tensor-ops.ts`](../../src/core/tensor-ops.ts): `exp(tensor)` y `log(tensor)` aplican la
  función elemento a elemento y registran su derivada cuando el tensor pide gradiente.
- [`test/core/unary-ops.test.ts`](../../test/core/unary-ops.test.ts): que son inversas y que el
  gradiente es el esperado.

## 6. Verificación

- `log(exp(x))` devuelve `x` dentro de la tolerancia de `f32`.
- El gradiente de `exp` es `e^x` y el de `log` es `1/x`, comprobados contra la fórmula.
- Se encadenan con `sum` y `mul` sin tocar el autograd.

## 7. Límites y errores comunes

- **`log(0) = -Infinity`** y `log` de un negativo es `NaN`; el dominio es $x > 0$.
- **Desbordamiento**: `exp(x)` se va a `Infinity` para $x$ grande; en softmax se resta el máximo para
  evitarlo (llega con softmax y la entropía cruzada).
- **Precisión `f32`**: en las colas, `exp` y `log` pierden precisión.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 3 y 4 (probabilidad e información).
- El truco *log-sum-exp* para estabilidad numérica.
