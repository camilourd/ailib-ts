# Alternativas a backpropagation

## 1. Qué problema resuelve

Backpropagation es exacto y eficiente, pero pide tres cosas: guardar las activaciones (memoria), un
grafo global (sincronía) y una pérdida diferenciable. No es biológicamente plausible —necesita
transportar los pesos y un error global— y no sirve cuando la señal de aprendizaje es una recompensa.

Esta nota es un **mapa**: qué familias existen, qué pieza de backprop cambian y qué se puede esperar de
ellas. Todavía no acompaña código; alimenta F3 (evolución) y F8 (ideas propias).

## 2. Definición: qué se puede cambiar

Entrenar con backprop es: estimar $\partial L/\partial \theta$, propagar ese gradiente hacia atrás y
actualizar. Una alternativa cambia al menos una de esas piezas:

- **cómo se estima** el gradiente, sin backward;
- **cómo se propaga**, con reglas locales o feedback aleatorio;
- **si hay gradiente**, buscando en el espacio de parámetros;
- **si hay pérdida**, usando una recompensa.

## 3. Las familias

### 3.1 Estimar el gradiente sin backward

- **Diferencias finitas / SPSA**: perturbar parámetros y medir el cambio de la pérdida.
- **Estrategias evolutivas (ES)**: muestrear perturbaciones, quedarse con las que mejoran y estimar el
  gradiente con ellas. Paraleliza muy bien y usa poca memoria; es **muy ineficiente en muestras**.
- **Forward gradient**: derivada direccional con una proyección aleatoria, sin backward. Barata en
  memoria; funciona en problemas pequeños, no compite a escala.

### 3.2 Evitar el backward con reglas locales

- **Feedback alignment**: matrices de retroalimentación aleatorias en vez de la transpuesta; evita el
  transporte de pesos.
- **Predictive coding**: cada capa predice la de arriba y aprende de su error local.
- **Equilibrium propagation**: redes de energía con dos fases (libre y fijada); el gradiente emerge de
  la diferencia entre ambas.
- **Target propagation**: cada capa recibe un «objetivo» en vez de un gradiente.
- **Forward-Forward**: dos pasadas, positiva y negativa, con una «bondad» local por capa; sin backward
  ni simetría de pesos.

### 3.3 Sin gradiente

- **Neuroevolución**: algoritmos genéticos, NEAT, CMA-ES; evolucionar pesos o arquitecturas.
- **Reservoir computing / ELM**: capa oculta aleatoria y fija; solo se entrena la salida (regresión
  lineal). Muy barato, poca capacidad.
- **Búsqueda directa**: Nelder-Mead, descenso por coordenadas; solo para problemas pequeños.

### 3.4 Sin pérdida diferenciable

- **Aprendizaje por refuerzo**: REINFORCE, actor-critic, PPO. El gradiente se estima con la recompensa;
  es lo que se usa cuando no hay pérdida derivable (o para RLHF).

### 3.5 Online y secuencial

- **RTRL, UORO, e-prop**: gradientes para redes recurrentes con trazas de elegibilidad; cambian
  memoria y tiempo por actualizaciones online.

### 3.6 Dentro del gradiente

No son alternativas, pero conviene conocerlas: **BPTT** (backprop a través del tiempo),
**checkpointing** (menos memoria a cambio de recomputar) e **implicit differentiation** (redes de
punto fijo, DEQ).

## 4. Ejemplo a mano

La idea más simple sin backward es medir la función. Para $f(x) = x^2$ en $x = 3$, con $\varepsilon = 0.1$:

$$\frac{f(3.1) - f(2.9)}{2\varepsilon} = \frac{9.61 - 8.41}{0.2} = 6 \approx 2x.$$

Es la derivada numérica que ya usamos en el gradcheck. **SPSA** y **ES** generalizan la idea a muchas
dimensiones: perturban en direcciones aleatorias y promedian $\frac{f(x+\sigma\varepsilon)\,\varepsilon}{\sigma}$.

## 5. Implementación (prevista)

- **F3**: evolución (HAEA, ES, CMA-ES) y comparación contra SGD en tareas pequeñas.
- **F8**: forward-forward, feedback alignment y forward gradient como experimentos propios.
- **F4**: e-prop cuando lleguen las RNN.
- Protocolo común: misma tarea y arquitectura, mismo presupuesto de muestras y de tiempo, varias
  semillas, y memoria/tiempo medidos.

## 6. Cómo se comparan

No basta con «funciona»: misma red, mismo dataset, mismo número de evaluaciones de la pérdida, varias
semillas, media y dispersión, y **backprop como baseline**. Se reportan calidad, muestras usadas,
memoria y tiempo.

Qué invalidaría una comparación: darle a un método más evaluaciones, una sola semilla o una tarea
distinta.

## 7. Límites y errores comunes

- Casi todas son **aproximaciones**: pueden funcionar en redes pequeñas y degradarse al crecer.
- La ineficiencia en muestras suele ser el factor decisivo, no la elegancia.
- «Biológicamente plausible» no significa «mejor»; backprop sigue siendo el baseline.
- Forward-forward y equilibrium propagation no compiten hoy a escala de LLM.
- Para LLM, el entrenamiento de pesos sigue siendo backprop + Adam; las alternativas viven en RL,
  ajuste con recompensas y experimentos.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Salimans et al. (2017), *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*.
- Baydin et al. (2022), *Gradients without Backpropagation*.
- Lillicrap et al. (2016), *Random synaptic feedback weights support error backpropagation*.
- Scellier y Bengio (2017), *Equilibrium Propagation*.
- Hinton (2022), *The Forward-Forward Algorithm*.
- Bellec et al. (2020), *e-prop* (aprendizaje online en RNN).
- Williams (1992), *REINFORCE*; Jaeger (2001), *reservoir computing*.
