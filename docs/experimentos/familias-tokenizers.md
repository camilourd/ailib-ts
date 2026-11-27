# Familias de tokenizers: F11

## WordPiece

Implementado en `src/data/wordpiece.ts`:

- vocabulario inicial de caracteres y continuaciones `##`;
- merges por ganancia de verosimilitud `freq(par)/(freq(izq)·freq(der))`;
- frecuencia mínima para evitar piezas raras;
- greedy longest-match;
- fallback de bytes para texto no visto;
- decode exacto.

Pruebas: 3 nuevas, 193/193.

### Resultado 2.048

WordPiece entrena 2.048 piezas + 256 bytes = **2.304** de vocabulario. En el arnés F10:

| Tokenizer | Vocab | Tokens/car. | F1 | ED | BLEU | BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | **0,284** | 0,540 | 0,567 | 0,309 | **3,068** |
| WordPiece 2048 | 2.304 | 0,614 | 0,374 | 0,774 | 0,126 | 5,544 |

**WordPiece es mucho peor en este presupuesto.** La razón principal es el tamaño: BERT usa ~30k
piezas; con 2k la ganancia de verosimilitud favorece piezas raras y deja palabras comunes en
caracteres. No es un fallo de exactitud: decode y encode son lossless.

### Hipótesis para mejorarlo

1. Subir el vocabulario a 8k/16k para comparar con BPE 2k **no sería justo**, pero sí sirve para ver
   la curva.
2. Cambiar el criterio de merge a frecuencia pura como ablación.
3. Añadir `##` a la puntuación/espacios.
4. Comparar con la implementación de HuggingFace para ver si el criterio se desvía.

## Unigram

Implementado en `src/data/unigram.ts`:

- candidatos por substrings;
- EM de probabilidades;
- Viterbi;
- fallback de bytes;
- decode exacto.

Pruebas: 3 nuevas, 196/196.

| Tokenizer | Vocab | Tokens/car. | F1 | ED | BLEU | BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | **0,284** | **0,540** | **0,567** | **0,309** | **3,068** |
| WordPiece 2048 | 2.304 | 0,614 | 0,374 | 0,774 | 0,126 | 5,544 |
| Unigram 2048 | 2.304 | 0,732 | 0,354 | 0,793 | 0,088 | 6,783 |

Unigram también queda muy por detrás a 2k. La EM tiende a quedarse en caracteres cuando el
vocabulario es pequeño y los candidatos son ruidosos. Es exacto, pero necesita más vocabulario,
mejor poda y probablemente un corpus mayor.

## SentencePiece

Implementado en `src/data/sentencepiece.ts` como **SentencePiece-BPE**:

- flujo crudo;
- `▁` para espacios;
- escape reversible para el marcador literal;
- BPE por debajo;
- decode exacto.

Pruebas: 2 nuevas, 198/198.

| Tokenizer | Vocab | Tokens/car. | F1 | ED | BLEU | BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | **0,284** | **0,540** | **0,567** | **0,309** | **3,068** |
| SentencePiece-BPE | 2.048 | 0,285 | 0,490 | 0,609 | 0,275 | 3,088 |

SentencePiece queda **casi igual en compresión** que BPE, pero algo peor en frontera/ED/BPB. El flujo
crudo con `▁` no aporta ventaja en español narrativo a 2k; sí sería relevante para idiomas sin
espacios o para preservar espacios exactos.

## Byte-level / BLT

Byte-level puro se puede comparar como tokenizer. BLT real no: los parches dependen de entropía y de
un modelo; habría que comparar bytes/parche y BPB, no tokens estáticos.

## Plan

- F11.0 WordPiece.
- F11.1 Unigram.
- F11.2 SentencePiece.
- F11.3 Byte-level/BLT + informe.

## Límites

- WordPiece se entrenó con 2k piezas, muy por debajo de BERT.
- La comparación con BPE es a igual presupuesto, no a igual calidad.
- El fallback de bytes cambia el WordPiece estándar (lossless a propósito).
