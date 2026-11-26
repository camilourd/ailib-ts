# Evolución vs gradiente

## 1. Qué problema resuelve

¿Cuándo conviene evolucionar y cuándo usar el gradiente? La pregunta no se responde con una opinión:
hace falta el mismo problema, la misma arquitectura y el mismo presupuesto de evaluaciones. Aquí se
compara HAEA con SGD en XOR.

## 2. Definición y protocolo

- **Problema**: XOR, red `2 → 4 ReLU → 2` con 22 parámetros.
- **Fitness**: cross-entropy sobre las cuatro filas (minimizar).
- **Presupuesto**: 24.120 ejemplos por método.
  - HAEA: 30 individuos × 201 evaluaciones × 4 ejemplos.
  - SGD: 6.030 pasos × 4 ejemplos.
- **Semillas**: 10 por método.
- **Éxito**: acertar 4/4.
- **Umbral**: CE < 0.1; se cuentan los ejemplos hasta alcanzarlo.
- **Ablación**: HAEA adaptativo (premia/castiga operadores) contra HAEA con tasas fijas.

## 3. Por qué así

- Comparar por **ejemplos evaluados**, no por generaciones o pasos: cada evaluación cuesta lo mismo.
- Misma red y mismos datos para los tres métodos.
- La semilla se repite para que la diferencia no sea un golpe de suerte.
- El umbral CE < 0.1 separa «acertar por poco» de «estar seguro».

## 4. Resultado con 10 semillas

| Método | Éxito (4/4) | CE < 0.1 | Ejemplos hasta CE < 0.1 | CE final | Exactitud |
| --- | --- | --- | --- | --- | --- |
| SGD `lr=0.5` | 7/10 | 7/10 | 282 | 0.1303 | 0.93 |
| HAEA fijo | 10/10 | 10/10 | 8.412 | 0.0024 | 1.00 |
| HAEA adaptativo | 10/10 | 7/10 | 12.291 | 0.0844 | 1.00 |

Lectura:

- **El gradiente es mucho más eficiente en muestras**: cuando converge, necesita ~282 ejemplos frente
  a ~8.400 del HAEA fijo (unas 30 veces menos).
- **La evolución es más fiable en este presupuesto**: resuelve las 10 semillas; SGD se queda en 7/10,
  probablemente por inicializaciones con ReLU muertas o mínimos locales.
- **La adaptación no ayudó aquí**: el HAEA fijo llegó antes al umbral (8.412 vs 12.291) y con mejor CE
  final. El premio/castigo puede favorecer el cruce demasiado pronto en un genoma pequeño; la
  adaptación no es una mejora gratis.
- **No hay ganador absoluto**: si el gradiente está disponible, suele ser la primera opción; la
  evolución es una alternativa robusta cuando no hay gradiente o cuando el gradiente se atasca.

## 5. Implementación

- [`lab/evolution/xor.ts`](../../lab/evolution/xor.ts): codifica los 22 parámetros del MLP, corre
  SGD y HAEA (adaptativo y fijo) con 10 semillas y resume las métricas.
- [`src/evolution/`](../../src/evolution/): el núcleo HAEA.
- [`src/train/trainer.ts`](../../src/train/trainer.ts): el bucle de SGD.
- [`src/nn/losses.ts`](../../src/nn/losses.ts): la cross-entropy usada como fitness.

## 6. Verificación

- Los tres métodos usan 24.120 ejemplos por semilla.
- La tabla sale de 10 semillas y muestra éxito, umbral, CE final y exactitud.
- 107/107 pruebas y `verify.mjs` en verde.
- El resultado negativo de la adaptación se reporta tal cual; no se oculta.

## 7. Límites y errores comunes

- **Un solo problema pequeño**: 22 parámetros; HAEA puede lucir distinto en genomas grandes.
- **Un solo `lr` para SGD**: no se ajustó; otro valor cambiaría el éxito de SGD.
- **Una sola arquitectura y una sola inicialización**.
- **Presupuesto igual, pero no tiempo igual**: no se midió pared; la evolución puede ser más lenta.
- **Adaptación simplificada**: sin CAHAEA ni extinción; el resultado negativo aplica a esta versión.
- **Éxito por exactitud y umbral por CE**: son dos criterios; la tabla muestra ambos.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Gómez, J. (2009), *A Hybrid Adaptive Evolutionary Algorithm*.
- Salimans et al. (2017), *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 8 (comparación de optimizadores).
