# LM de caracteres

## 1. Qué problema resuelve

Un **modelo de lenguaje** aprende la probabilidad del siguiente token dado el contexto:

$$P(w_t \mid w_{<t}).$$

Entrenado con texto, puede continuar una frase. Aquí es un LM de caracteres byte-level: predice el
siguiente byte a partir de los anteriores.

## 2. Definición y notación

- **Contexto**: los tokens anteriores; en una RNN, el estado oculto los resume.
- **Pérdida**: cross-entropy media por token,

$$L = -\frac{1}{T}\sum_{t=1}^{T}\ln P(w_t \mid w_{<t}).$$

- **Perplejidad**: $e^{L}$; cuántas opciones efectivas maneja el modelo.
- **Generación**: se muestrea el siguiente token de la distribución, se añade al contexto y se repite.

## 3. Por qué así

- **Cada byte es un token**: vocabulario de 256, sin `<unk>`.
- **Embedding → RNN → Linear**: el embedding da un vector por byte; la RNN resume el contexto; la capa
  final da un logit por byte.
- **BPTT**: el backward atraviesa los 32 pasos de la ventana; no hay gradiente manual.
- **Temperatura**: divide los logits antes del softmax. Baja → más conservador; alta → más aleatorio.

## 4. Ejemplo a mano

Con la palabra `"hola"`, el modelo recibe `h, o, l` y aprende a predecir `o, l, a`. La pérdida es el
promedio de tres cross-entropies. Al generar, se parte de `"En un "` y se muestrea byte a byte.

## 5. Implementación

- [`src/models/char-lm.ts`](../../src/models/char-lm.ts): `CharLM` con `forward`, `loss` y `generate`;
  usa `Embedding`, `RNN` y `Linear`.
- [`lab/lm/run.ts`](../../lab/lm/run.ts): 6000 pasos sobre un texto de 792 bytes; `Embedding(32)`,
  `RNN(64)`, `SGD(0.05)`.
- [`lab/lm/texto.ts`](../../lab/lm/texto.ts): el texto de trabajo, original.
- [`test/models/char-lm.test.ts`](../../test/models/char-lm.test.ts): formas, pérdida, gradiente,
  generación y validaciones.

Con 6000 pasos la pérdida baja de **5.5710 a 0.7585**. Un ejemplo generado desde `"En un "`:

```
En un ajuselo recerda belpora, a veces un pada alo le piridado re prende cospiosto becorri,
lo que bus� liv�bze. soygua coteduunida. No oprampa la r�anciende una llot
```

Son fragmentos y palabras sueltas, no frases: el modelo tiene 64 unidades y 32 pasos de contexto. Los
`�` aparecen porque el LM byte-level puede emitir bytes que no forman UTF-8 válido.

## 6. Verificación

- `loss` devuelve un escalar y el gradiente llega a los seis tensores de parámetros.
- La generación devuelve bytes válidos del vocabulario y es reproducible por semilla.
- La pérdida baja de forma clara en el laboratorio.
- 123/123 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Byte-level**: 256 clases y posibles secuencias UTF-8 inválidas al generar; un tokenizer de
  caracteres o BPE lo mitiga.
- **Contexto corto**: 32 pasos; la RNN simple olvida lo lejano (el siguiente experimento lo mide).
- **Sin batch ni validación**: una sola secuencia; se puede sobreajustar.
- **Sin *schedules* ni *clipping***: la tasa es fija.
- **Modelo pequeño**: 64 ocultas; no se le puede pedir coherencia larga.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Bengio et al. (2003), *A Neural Probabilistic Language Model*.
- Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks*.
- Goodfellow, Bengio y Courville, *Deep Learning*, cap. 10.
