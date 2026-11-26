# LM transformer

## 1. Qué problema resuelve

El LM de caracteres de F4 usaba una RNN: procesa en orden y arrastra un estado. Un **transformer**
mira toda la secuencia con atención causal en paralelo; aquí se entrena uno pequeño para predecir el
siguiente byte.

## 2. Definición y notación

- **Tokens**: bytes 0..255 (tokenizer byte-level).
- **Embedding**: `[256, d_model]`.
- **Bloques**: cada uno `x + MultiHead(LN(x))` y `x + FFN(LN(x))`, con RoPE y máscara causal.
- **Norm final** y `Linear(d_model, 256)` para los logits.
- **Pérdida**: cross-entropy media de predecir cada token a partir del anterior.
- **Generación**: se muestrea el último logit, se añade al contexto y se repite.

## 3. Por qué así

- La atención causal permite que cada posición vea solo el pasado, igual que un LM autorregresivo.
- RoPE aporta la posición sin sumar embeddings absolutos.
- El residual + LayerNorm pre-norm mantiene el gradiente y la escala.
- Es la misma arquitectura del bloque, apilada; aquí con **un** bloque para que quepa en el
  presupuesto.

## 4. Ejemplo a mano

Entrada `[0, 1, 2, 3]`:

- Embedding: `[4, 32]`.
- Un bloque: `[4, 32]`; RoPE en Q/K; máscara causal.
- Norm final y salida: `[4, 256]`.
- Pérdida con los objetivos `[1, 2, 3]` (tres predicciones).
- En generación, para el token 80 se vuelve a pasar toda la secuencia; sin KV cache, es $O(L^2)$ por
  paso.

## 5. Implementación

- [`src/models/transformer-lm.ts`](../../src/models/transformer-lm.ts): `TransformerLM` con
  `forward`, `loss`, `generate` y `parameters`.
- [`src/nn/sampling.ts`](../../src/nn/sampling.ts): muestreo con temperatura compartido con `CharLM`.
- [`lab/transformer/lm.ts`](../../lab/transformer/lm.ts): 3000 pasos, un bloque, cuatro cabezas,
  `d_model=32`, `d_ff=64`.
- [`lab/lm/texto.ts`](../../lab/lm/texto.ts): el mismo texto de F4.

Con 3000 pasos la pérdida baja de **5.4488 a 1.2402** y el **BPC test es 5.092**. Una muestra:

```
En un acÛpadenr. pa ellires el lsate io o a esa, esex eniNesel e?. es o esntentes Sge
```

Es texto con fragmentos, no frases; el modelo es diminuto y el corpus también.

## 6. Verificación

- Los logits tienen forma `[secuencia, 256]`; la pérdida es escalar y el gradiente llega a todos los
  parámetros.
- La generación es reproducible por semilla y devuelve bytes del vocabulario.
- 160/160 pruebas y `verify.mjs` en verde.
- **Comparación honesta**: con el mismo texto y 3000 pasos, el LM de bytes con RNN daba un BPC de
  4.009; este transformer, 5.092. Un bloque de 32 dimensiones no compensa la falta de datos/pasos.

## 7. Límites y errores comunes

- **Un bloque, 32 dimensiones**: no es un transformer grande.
- **3000 pasos y 624 bytes de train**: insuficiente para competir con la RNN en este caso.
- **Sin dropout, sin batch, sin KV cache**: generación lenta y sobreajuste probable.
- **Byte-level**: puede emitir secuencias UTF-8 inválidas.
- **BPC peor que la RNN no invalida la arquitectura**: mide este presupuesto, no el techo del modelo.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Vaswani et al. (2017), *Attention Is All You Need*.
- Radford et al. (2019), *GPT-2* (LM autorregresivo con transformer).
- Karpathy, *nanoGPT* (implementación mínima).
