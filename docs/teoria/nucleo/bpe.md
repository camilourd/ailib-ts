# BPE

## 1. Qué problema resuelve

Con bytes, cada carácter acentuado ocupa 2 tokens y un emoji 4; las secuencias son largas. **BPE**
(*Byte Pair Encoding*) aprende a fusionar los pares de bytes más frecuentes y crea tokens que
representan trozos: `"baja"` puede pasar de 4 tokens a 1.

## 2. Algoritmo

1. Empieza con los 256 bytes.
2. Cuenta los pares adyacentes del corpus de entrenamiento.
3. Fusiona el par más frecuente en un token nuevo y lo añade al vocabulario.
4. Repite hasta el vocabulario objetivo o hasta que ningún par llegue a la frecuencia mínima.

`encode` aplica los merges en el orden aprendido; `decode` expande cada token a sus bytes.

## 3. Por qué comprime

- Un par frecuente deja de ocupar dos tokens.
- Tras varios merges, palabras frecuentes quedan en uno o pocos tokens.
- El vocabulario es fijo antes de entrenar el LM; BPE no se aprende con gradiente.

El precio: más vocabulario → embedding y salida más grandes; tokens raros con pocos datos; y el
reparto depende del corpus (un idioma o dominio con poca presencia comprime menos).

## 4. Experimento

Texto de 780 caracteres, split 624/156. BPE entrenado **solo con train**, vocabulario objetivo 512,
frecuencia mínima 2. El corpus solo permitió **334 tokens** (78 merges).

| Tokenizer | Vocabulario | Tokens/carácter | Pérdida/token | BPC test | Tokens test |
| --- | --- | --- | --- | --- | --- |
| Bytes | 256 | 1.019 | 1.7498 | 4.009 | 156 |
| BPE | 334 | 0.514 | 1.9457 | 5.627 | 104 |

BPE reduce los tokens a **la mitad** (0.514 por carácter), pero a igual número de pasos su **BPC es
peor**: 5.627 frente a 4.009. La pérdida por token también es mayor (1.946 vs 1.750).

Lectura honesta:

- **La compresión funciona**: la mitad de tokens y, a igual ventana de 32 tokens, el doble de
  caracteres de contexto.
- **El LM no mejora solo por comprimir**: BPE tiene 334 clases de salida, más parámetros y menos
  ejemplos por token en un corpus diminuto; 3000 pasos no le bastan.
- **No es gratis**: se cambia longitud por vocabulario. En F5, con atención $O(L^2)$, reducir la
  longitud sí será una ventaja grande; aquí el cuello es el ajuste.

## 5. Implementación

- [`src/data/bpe.ts`](../../src/data/bpe.ts): `BPETokenizer` con `train`, `encode`, `decode`,
  `vocabularySize` y `merges`.
- [`test/data/bpe.test.ts`](../../test/data/bpe.test.ts): merges, compresión, ida y vuelta con
  acentos/emoji, determinismo y validaciones.
- [`lab/lm/bpe.ts`](../../lab/lm/bpe.ts): split train/test, BPE sobre train, dos LMs y BPC.

## 6. Verificación

- `"baja baja baja"` comprime y vuelve intacto.
- Acentos y emoji sobreviven a `decode(encode(texto))`.
- La misma entrada produce los mismos merges.
- Tokens fuera del vocabulario y parámetros inválidos fallan.
- 134/134 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Corpus de 624 caracteres**: el vocabulario se queda en 334 y los merges son locales.
- **Mismos 3000 pasos, no mismo número de caracteres vistos**: la comparación es de presupuesto de
  pasos, no de convergencia.
- **BPC, no perplejidad**: perplejidad por token no es comparable entre vocabularios; BPC sí.
- **Sin atención todavía**: la ventaja de contexto se verá en F5.
- **Sin guardar los merges en un checkpoint**: el tokenizer habrá que persistirlo junto al modelo.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Sennrich, Haddow y Birch (2016), *Neural Machine Translation of Rare Words with Subword Units*.
- Radford et al. (2019), *GPT-2* (BPE byte-level con vocabulario grande).
- Documentación de `tokenizers` (Hugging Face) para BPE/WordPiece/Unigram.
