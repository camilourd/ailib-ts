# SGD

## 1. Qué problema resuelve

El autograd dice **hacia dónde** baja la pérdida (el gradiente), pero no **cuánto** moverse. SGD es la
regla más simple para usar ese gradiente: dar un paso pequeño en la dirección opuesta.

## 2. Definición y notación

$$\theta \leftarrow \theta - \eta \, \nabla L(\theta)$$

- $\theta$: el conjunto de parámetros (pesos y sesgos).
- $\nabla L(\theta)$: el gradiente de la pérdida, calculado por el autograd.
- $\eta$: la **tasa de aprendizaje** (*learning rate*).

Se llama **estocástico** porque el gradiente no se calcula con todo el dataset, sino con un **lote**
(una muestra): es una estimación ruidosa del gradiente real. En esta pieza el lote es todo el dato
disponible; el ruido llega cuando aparezca el batching.

## 3. Derivación y por qué funciona

Si el gradiente es $g$ y se da un paso $\eta g$, la aproximación de primer orden de Taylor dice:

$$L(\theta - \eta g) \approx L(\theta) - \eta\,\|g\|^2.$$

Para $\eta$ suficientemente pequeño, el término $\eta\|g\|^2$ domina y la pérdida baja. La condición
general es que $\eta$ sea menor que $2/L$, donde $L$ es la constante de Lipschitz del gradiente; en
funciones no convexas no hay garantía de llegar al mínimo global.

Para $f(x) = x^2$, el gradiente es $2x$ y la actualización queda:

$$x \leftarrow x - \eta(2x) = (1 - 2\eta)x.$$

Converge si $|1-2\eta| < 1$, es decir, $0 < \eta < 1$. Con $\eta \ge 1$ oscila o diverge; con $\eta$
muy pequeño tarda.

## 4. Ejemplo a mano

Con $x_0 = 4$ y $\eta = 0.1$, el factor por paso es $0.8$:

| Paso | $x$ | $f(x)$ |
| --- | --- | --- |
| 0 | 4.0000 | 16.0000 |
| 1 | 3.2000 | 10.2400 |
| 2 | 2.5600 | 6.5536 |
| 5 | 1.3107 | 1.7180 |
| 10 | 0.4295 | 0.1845 |
| 20 | 0.0461 | 0.0021 |

## 5. Implementación

- [`src/optim/sgd.ts`](../../src/optim/sgd.ts): `Optimizer` (contrato `step`/`zeroGrad`) y `SGD`, que
  valida la tasa de aprendizaje y actualiza en el sitio los parámetros que tienen gradiente.
- [`test/optim/sgd.test.ts`](../../test/optim/sgd.test.ts): convergencia de la cuadrática, varios
  parámetros, `zeroGrad`, parámetro sin gradiente y tasas inválidas.
- [`lab/sgd/run.ts`](../../lab/sgd/run.ts): imprime la convergencia de $x^2$ paso a paso.

## 6. Verificación

- La cuadrática $x^2$ desde $x=4$ queda a menos de $10^{-6}$ del mínimo tras 100 pasos.
- Cada parámetro se mueve con **su** gradiente, no con el de otro.
- `zeroGrad` deja el gradiente en `null` para que el siguiente paso no acumule.
- Un parámetro sin gradiente no se toca.
- Una tasa de aprendizaje no positiva o no finita falla.

Qué fallaría con el signo cambiado: $x$ se alejaría del mínimo y la pérdida crecería en cada paso.

## 7. Límites y errores comunes

- **Tasa fija**: sin *schedules* (decaimiento, calentamiento) todavía.
- **Sin momentum ni Adam**: son optimizadores distintos, llegarán después.
- **Sin *weight decay*** ni otras regularizaciones.
- **Lote completo**: el ruido del gradiente estocástico y el batching llegan con el bucle de
  entrenamiento.
- **Sensible a $\eta$**: demasiado grande diverge; demasiado pequeña tarda.
- **Sin garantías en no convexas**: puede quedarse en un mínimo local o en una meseta.
- **Solo `f32`**.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Robbins y Monro (1951), *A Stochastic Approximation Method*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 8 (optimización).
- PyTorch, documentación de `torch.optim.SGD` (momentum, weight decay).
