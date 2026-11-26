# SFT

## 1. Qué problema resuelve

El preentreno aprende a continuar texto, no a responder preguntas. **SFT** (*supervised fine-tuning*)
ajusta el modelo con pares instrucción→respuesta para que asocie la pregunta con una respuesta corta.

## 2. Definición y notación

- **Par**: `Pregunta: ...` y `Respuesta: ...`.
- **Formato**: `Pregunta: {instrucción}\nRespuesta: {respuesta}\n`.
- **Pérdida**: la misma cross-entropy del LM sobre la secuencia formateada.
- **Antes/después**: se evalúa el mismo conjunto de validación antes y después de ajustar.

No se enmascara el prompt en esta primera versión: la pérdida incluye pregunta y respuesta. Es una
limitación consciente, no un olvido.

## 3. Por qué funciona

- El preentreno deja un modelo de lenguaje; el SFT le muestra **qué formato** y **qué respuestas** se
  esperan.
- Con pocos pares se adapta el estilo, no se aprenden hechos nuevos.
- Comparar antes/después separa «mejor pérdida» de «mejor generación».

## 4. Datos y presupuesto

- 16 pares de train y 6 de validación, escritos a mano sobre el comienzo del Quijote.
- Modelo preentrenado: semilla 1, paso 1000, 1.401.792 parámetros.
- 300 pasos, `lr=0.0005`, clipping 1.0, batch 1.
- Evaluación: pérdida media de los 6 pares de validación; perplejidad = $e^{\mathcal{L}}$.

| Métrica | Antes | Después |
| --- | --- | --- |
| Pérdida de validación | 6.0114 | **5.7564** |
| Perplejidad | 408.05 | **316.20** |
| Pérdida de train (paso 299) | 5.9352 | 5.6202 |

La pérdida baja, pero las muestras siguen siendo fragmentos. El SFT mejora el ajuste a los pares, no
convierte al modelo en un asistente: 16 pares y 300 pasos no alcanzan.

## 5. Implementación

- [`lab/transformer/sft.ts`](../../lab/transformer/sft.ts): carga el checkpoint, mide antes, ajusta y
  mide después.
- [`data/sft-quijote.json`](../../data/sft-quijote.json): pares de train y validación.
- [`src/models/transformer-lm.ts`](../../src/models/transformer-lm.ts): `loadParameters` copia el
  checkpoint por nombre.
- [`data/sft-report.json`](../../data/sft-report.json): pérdidas, perplejidades y muestras.

## 6. Verificación

- El modelo carga el checkpoint y `loadParameters` exige todas las claves y formas.
- Antes y después usan los mismos 6 pares.
- 168/168 pruebas y `verify.mjs` en verde.
- El resultado negativo se conserva: la generación no mejora de forma clara.

## 7. Límites y errores comunes

- **16 pares**: insufficientes para un asistente.
- **Pérdida sobre toda la secuencia**: el prompt pesa tanto como la respuesta.
- **Sin máscara de respuesta**: no se optimiza solo lo que se genera.
- **Sin plantilla de chat**: el formato es texto plano.
- **Un checkpoint de 1000 pasos**: el preentreno tampoco estaba terminado.
- **Perplejidad baja no es calidad**: mirar siempre las muestras.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Ouyang et al. (2022), *Training language models to follow instructions with human feedback*.
- Wei et al. (2021), *Finetuned Language Models Are Zero-Shot Learners*.
