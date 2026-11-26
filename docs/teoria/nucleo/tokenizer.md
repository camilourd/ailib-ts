# Tokenizer

## 1. Qué problema resuelve

Un modelo no lee texto: lee números. El **tokenizer** convierte una cadena en una secuencia de tokens
y sabe volver atrás. Sin esa pieza, el resto del LM no tiene entrada.

## 2. Definición y notación

- **Byte**: un valor 0..255.
- **UTF-8**: codificación que representa cada carácter Unicode como 1–4 bytes.
- **Vocabulario**: los 256 bytes posibles.
- `encode(texto) → number[]` y `decode(tokens) → texto`.

## 3. Por qué byte-level

- **Reversible**: cualquier cadena UTF-8 válida vuelve a ser la misma.
- **Sin «desconocido»**: todo texto cae en los 256 bytes; no hay token `<unk>`.
- **Simple**: no hay que entrenar un vocabulario.
- **Precio**: un carácter acentuado ocupa 2 bytes y un emoji 4; las secuencias son más largas que con
  un tokenizer de caracteres o BPE.

La alternativa clásica es **BPE** (subpalabras), que aprende trozos frecuentes y acorta las secuencias;
llega después, cuando el LM lo necesite.

## 4. Ejemplo a mano

- `"hola"` → `[104, 111, 108, 97]`.
- `"ñ"` → `[195, 177]` (dos bytes).
- `"🐢"` → cuatro bytes.
- `"hola, ñandú 🐢"` → 18 tokens.

## 5. Implementación

- [`src/data/tokenizer.ts`](../../src/data/tokenizer.ts): `Tokenizer` y `ByteTokenizer` con
  `encode`/`decode`; `decode` valida que cada token sea un byte.
- [`test/data/tokenizer.test.ts`](../../test/data/tokenizer.test.ts): ASCII, acentos, emoji, cadena
  vacía y tokens inválidos.
- [`lab/embedding/run.ts`](../../lab/embedding/run.ts): tokeniza y muestra los primeros bytes.

## 6. Verificación

- `decode(encode(texto)) === texto` para ASCII, acentos y emoji.
- `"hola"` da exactamente `[104, 111, 108, 97]`; `"ñ"` da `[195, 177]`.
- Un token fuera de 0..255, negativo o no entero falla con `RangeError`.
- 114/114 pruebas y `verify.mjs` en verde.

## 7. Límites y errores comunes

- **Sin BPE ni tokens especiales**: todo es un byte; no hay `<inicio>` ni `<fin>`.
- **Secuencias largas**: el texto acentuado y los emojis cuestan más tokens.
- **UTF-8 inválido**: al decodificar bytes sueltos puede aparecer el carácter de reemplazo.
- **Sin normalización Unicode**: `"é"` compuesto y precompuesto son secuencias distintas.

## 8. Mi explicación

Pendiente: la escribe la persona, sin mirar el resto de la nota.

## 9. Para profundizar (opcional)

- Especificación de UTF-8 (RFC 3629).
- Sennrich, Haddow y Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* (BPE).
- Documentación de `TextEncoder`/`TextDecoder` en Node.
