# Tokenizadores

## 1. Qué problema resuelve

Un tokenizer decide qué unidades ve el modelo. No es neutro: cambia la longitud de la secuencia, el
vocabulario, el coste y lo que el modelo puede aprender. La perplejidad por token no se puede comparar
entre tokenizers; sí **bits por byte (BPB)** o bits por carácter.

## 2. Familias comparadas

| Tokenizer | Unidad | Vocabulario | OOV |
| --- | --- | ---: | --- |
| byte | byte UTF-8 | 256 | no |
| carácter | code point | 77 (Quijote) | `<unk>` |
| BPE 512 | subpalabra | 512 | no (bytes) |
| BPE 1024 | subpalabra | 1.024 | no |
| BPE 2048 | subpalabra | 2.048 | no |
| palabra | palabra/signo/espacio | 2.228 | `<unk>` |

## 3. Protocolo

- Quijote: 180.000 caracteres de train y 20.001 de validación.
- Mismo transformer: `d_model=64`, 4 cabezas, 2 bloques, `d_ff=256`.
- 600 pasos, contexto 96 tokens, `lr=0.005`, semilla 1.
- Métricas: tokens/carácter, **BPB** y **bits/carácter**; la perplejidad se reporta pero no se compara
  entre vocabularios.
- Mismo número de pasos y tokens de contexto, no los mismos caracteres por ventana.

## 4. Resultados

| Tokenizer | Vocab | Params | Tokens/carácter | BPB | Bits/carácter | PPL/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| byte | 256 | 133.120 | 1,021 | 4,831 | 4,919 | 28,5 |
| carácter | 77 | 110.029 | 1,000 | 4,282 | 4,360 | 20,5 |
| BPE 512 | 512 | 166.144 | 0,434 | 3,579 | 3,644 | 337,6 |
| BPE 1024 | 1.024 | 232.192 | 0,331 | 3,149 | 3,207 | 817,5 |
| BPE 2048 | 2.048 | 364.288 | 0,265 | 2,851 | 2,902 | 1.696,8 |
| **palabra** | **2.228** | 387.508 | 0,384 | **2,035** | **2,072** | 42,0 |

## 5. Lectura

- **Byte es el peor**: gasta un token por byte y no aprovecha caracteres multibyte.
- **Carácter mejora a byte** en BPB (4,282 vs 4,831) con un vocabulario pequeño.
- **BPE mejora al crecer el vocabulario**: 3,579 → 3,149 → 2,851. Más merges, menos tokens por carácter.
- **Palabra gana en BPB** (2,035) con 2.228 tokens y un modelo de tamaño parecido a BPE 2048.
  La secuencia es más corta en unidades semánticas, aunque queden OOV.
- La **perplejidad por token no es comparable**: byte tiene PPL 28,5 y BPB 4,831; palabra tiene PPL
  42,0 y BPB 2,035. El vocabulario cambia la base de la exponencial.
- Un corpus limpio y pequeño favorece palabra; un vocabulario abierto/multilingüe favorece subpalabras
  con fallback a bytes.

## 6. Implementación

- [`src/data/char.ts`](../../src/data/char.ts): tokenizer por code points.
- [`src/data/word.ts`](../../src/data/word.ts): palabras, números, signos y espacios.
- [`src/data/bpe.ts`](../../src/data/bpe.ts): BPE existente.
- [`lab/lm/tokenizers.ts`](../../lab/lm/tokenizers.ts): comparación completa.
- [`data/tokenizers-report.json`](../../data/tokenizers-report.json): resultados crudos.
- Tests: `test/data/char.test.ts` y `test/data/word.test.ts`.

## 7. Verificación

- Ida y vuelta con acentos y emoji en carácter.
- Palabra separa signos/espacios y reconstruye; `<unk>` para lo no visto.
- Mismo modelo, pasos, contexto y semilla para todos.
- 182/182 pruebas, build y lint en verde.

## 8. Límites

- Un corpus, un modelo pequeño y 600 pasos: no es una ley general.
- La palabra tiene OOV; no se midió la tasa de `<unk>`.
- El contexto de 96 tokens cubre distinto número de caracteres según tokenizer.
- El vocabulario cambia el número de parámetros; se reporta, no se iguala.
- No se probaron WordPiece ni Unigram; quedan como siguiente comparación.

## 9. Conclusión

Para este presupuesto, **palabra > BPE 2048 > BPE 1024 > BPE 512 > carácter > byte** en BPB. La
decisión práctica no es «el tokenizer más pequeño» ni «el vocabulario más grande», sino el que da más
información por byte con un coste de secuencia razonable.
