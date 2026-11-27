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

Byte-level puro (`byte-256`) y un **patcher por entropía** BLT-style:

- n-gram de bytes para estimar entropía;
- agrupa bytes mientras la entropía del siguiente byte es baja;
- vocabulario de parches + fallback de bytes;
- decode exacto.

| Tokenizer | Vocab | Tokens/car. | F1 | ED | BLEU | BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | **0,284** | **0,540** | **0,567** | **0,309** | **3,068** |
| Byte-256 | 256 | 1,018 | 0,300 | 0,914 | 0,009 | 6,748 |
| Patcher entropía 2 | 963 | 0,781 | 0,276 | 0,912 | 0,014 | 6,843 |
| Patcher entropía 3 | 2.048 | 0,490 | 0,273 | 0,913 | 0,015 | 5,110 |

El patcher reduce tokens frente a bytes puros y mejora BPB, pero **sigue lejos de BPE**. No es BLT
completo: BLT real aprende un modelo de parches y compara bytes/parche, no tokens estáticos. Aquí solo
probamos la intuición de “parchear por entropía”.

Pruebas: 2 nuevas, 200/200.

## F11.4 comparación GPU (~5,8M)

Para igualar condiciones:

- mismo corpus de **50.000 caracteres** para entrenar los tokenizers;
- mismo test del Quijote y Lazarillo externo;
- mismo transformer: `d_model=256`, 6 bloques, 8 cabezas, `d_ff=1024`, contexto 128;
- mismo batch 8, 400 pasos, semilla 1;
- vocabulario objetivo 2.048.

El modelo quedó en **5,79M parámetros** (no 8M): el vocabulario 2k reduce las matrices de salida.

| Tokenizer | Vocab real | BPB test | BPB externo |
| --- | ---: | ---: | ---: |
| BPE 2048 | 2.048 | 2,922 | 3,161 |
| SentencePiece-BPE | 2.048 | 2,913 | 3,170 |
| WordPiece | 1.604 | **2,639** | **2,893** |
| Unigram | 2.304 | **2,624** | **2,800** |

**Cambio de resultado:** a ~5,8M, WordPiece y Unigram **ganan BPB** a BPE, aunque sigan teniendo peor
tokens/carácter, frontera y ED. Es la señal contraria al proxy de 32 dimensiones.

### Réplica parcial (semilla 2)

| Tokenizer | BPB test | BPB externo |
| --- | ---: | ---: |
| BPE 2048 | 2,895 | 3,137 |
| SentencePiece-BPE | 2,924 | 3,170 |

BPE y SentencePiece son **estables** entre semillas. WordPiece y Unigram no se replicaron: a 400
pasos cada corrida tarda demasiado en la GTX 1060 de 3 GB.

**8M real no ejecutado.** El modelo de esta comparación es 5,79M. Para 8M haría falta subir bloques y
`d_ff`; en 3 GB es posible pero muy lento.

Lectura provisional:

- La granularidad fina puede ayudar al LM pequeño a predecir bytes, aunque comprima peor.
- WordPiece no alcanzó 2.048 piezas en 50k (quedó en 1.604); Unigram usa 2.304 con bytes.
- Es **una sola semilla** para WordPiece/Unigram; hay que replicar antes de concluir.
- El modelo es 5,8M, no 8M; falta escalar.

## F11.5 híbridos y ganador evolutivo

Mismo montaje que F11.4: 50k, 400 pasos, batch 8, d256/6/1024 (~5,8M), semilla 1.

| Tokenizer | Vocab | BPB test | BPB externo |
| --- | ---: | ---: | ---: |
| BPE 2048 | 2.048 | 2,922 | 3,161 |
| BPE + sufijos | 2.085 | 2,910 | 3,153 |
| BPE + prefijos + sufijos | 2.098 | 2,902 | 3,146 |
| BPE + prefijos + sufijos + números | 2.098 | 2,902 | 3,146 |
| SentencePiece-BPE | 2.048 | 2,913 | 3,170 |
| **WordPiece** | 1.604 | **2,639** | **2,893** |
| **Unigram** | 2.304 | **2,624** | **2,800** |

- El ganador evolutivo de F10.3 era `prefijo+sufijo`; en GPU **mejora a BPE** 0,020/0,015, pero no
  se acerca a WordPiece/Unigram.
- Los números no cambian nada: mismo vocabulario y mismos tokens.
- WordPiece/Unigram siguen ganando BPB **con peor compresión**.
- Una semilla y 5,8M; falta replicar y escalar.

## F11.6a barrido de vocabulario

Train completo del Quijote (70%), sin LM. Vocabulario objetivo 2k/8k/16k.

| Tokenizer | Objetivo | Real | Tokens/car. | F1 | ED |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPE | 2.048 | 2.048 | 0,284 | 0,540 | 0,567 |
| BPE | 8.192 | **6.190** | 0,231 | **0,598** | **0,472** |
| BPE | 16.384 | **6.190** | 0,231 | **0,598** | **0,472** |
| SentencePiece | 2.048 | 2.048 | 0,285 | 0,490 | 0,609 |
| SentencePiece | 8.192 | 6.193 | 0,233 | 0,526 | 0,532 |
| WordPiece | 2.048 | 2.304 | 0,614 | 0,374 | 0,774 |
| WordPiece | 8.192 | **3.116** | 0,466 | 0,381 | 0,684 |
| WordPiece | 16.384 | **3.116** | 0,466 | 0,381 | 0,684 |
| Unigram | 2.048 | 2.304 | 0,732 | 0,354 | 0,793 |
| Unigram | 8.192 | 8.448 | 0,574 | 0,350 | 0,765 |
| Unigram | 16.384 | 16.640 | 0,532 | 0,303 | 0,817 |

Consecuencias:

- **El vocabulario objetivo no se alcanza**: BPE se queda en 6.190, WordPiece en 3.116,
  SentencePiece en 6.193. El corpus de 129k no da para más con estos algoritmos.
- Subir vocabulario **sí mejora** BPE, SentencePiece y WordPiece en compresión/frontera.
- Unigram crece hasta 16.640, pero **empeora F1/ED**; su EM no está bien calibrada.
- BPE 6.190 es el mejor en tokens/carácter, frontera y ED.
- WordPiece sigue por detrás incluso con 3.116 piezas.

Esto refuerza que la comparación anterior a 2k no era justa para WordPiece/Unigram, pero tampoco
aparece una ventaja de ellos en estas métricas.

## F11.6b variaciones del LM

Primera variación: **1,9M** (`d_model=160`, 4 bloques, `d_ff=640`), 100 pasos, contexto 128.
No es comparable con los 400 pasos de 5,8M, pero sirve para ver la dirección.

| Tokenizer | Params | BPB test | BPB externo |
| --- | ---: | ---: | ---: |
| BPE 2048 | 1.894.848 | 2,914 | 3,104 |
| BPE + sufijo | 1.906.725 | 2,922 | 3,112 |
| BPE + prefijo + sufijo | 1.910.898 | **2,913** | **3,101** |
| BPE + prefijo + sufijo + número | 1.910.898 | **2,913** | **3,101** |
| WordPiece | 1.752.324 | 2,953 | 3,096 |
| Unigram | 1.977.024 | 3,097 | 3,211 |

- A 1,9M/100 pasos, WordPiece y Unigram **no ganan**; Unigram es el peor.
- El híbrido BPE+prefijo+sufijo queda casi igual que BPE; el sufijo solo empeora.
- La ventaja de WordPiece/Unigram en 5,8M/400 pasos **depende del presupuesto**: no aparece aquí.

### Variación de contexto

Mismo 5,8M, 100 pasos, contexto 64:

| Tokenizer | BPB test | BPB externo |
| --- | ---: | ---: |
| BPE 2048 | 2,916 | 3,113 |
| BPE + sufijo | 2,908 | 3,092 |
| BPE + prefijo + sufijo | **2,893** | **3,087** |
| BPE + prefijo + sufijo + número | **2,893** | **3,087** |
| WordPiece | 2,856 | 3,020 |
| Unigram | 2,962 | 3,075 |

Comparación de presupuestos:

| Tokenizer | 1,9M/100 | 5,8M/100 ctx64 | 5,8M/400 ctx128 |
| --- | ---: | ---: | ---: |
| BPE | 2,914 / 3,104 | 2,916 / 3,113 | 2,922 / 3,161 |
| BPE + sufijo | 2,922 / 3,112 | 2,908 / 3,092 | 2,910 / 3,153 |
| BPE + prefijo + sufijo | 2,913 / 3,101 | 2,893 / 3,087 | 2,902 / 3,146 |
| BPE + prefijo + sufijo + número | 2,913 / 3,101 | 2,893 / 3,087 | 2,902 / 3,146 |
| WordPiece | 2,953 / 3,096 | 2,856 / 3,020 | **2,639 / 2,893** |
| Unigram | 3,097 / 3,211 | 2,962 / 3,075 | **2,624 / 2,800** |

- El híbrido BPE mejora a BPE en las tres configuraciones, pero poco.
- WordPiece/Unigram mejoran al pasar de 100 a 400 pasos: **la ventaja depende del presupuesto de optimización**, no solo de la escala.
- Contexto 64 cambia poco el ranking.

### Alternativas al transformer clásico

Mismo 5,8M, 100 pasos, contexto 128, BPE y BPE+prefijo+sufijo:

| Arquitectura | Params | BPE test/ext | Híbrido test/ext |
| --- | ---: | ---: | ---: |
| Pre-norm | 5.789.696 | 2,859 / 3,058 | **2,837 / 3,040** |
| Post-norm | 5.789.696 | **2,855 / 3,054** | 2,835 / 3,040 |
| GRU (2 capas, 768) | 8.006.656 | 2,902 / 3,085 | 2,900 / 3,090 |
| LSTM (2 capas, 512) | 5.253.120 | 2,918 / 3,097 | 2,921 / 3,107 |

- Pre-norm y post-norm son **casi idénticos**.
- GRU y LSTM **no superan** al transformer; LSTM es el peor.
- El híbrido BPE mejora al BPE también en recurrentes, pero no alcanza al transformer.
- Parámetros no igualados: GRU tiene 8M y LSTM 5,25M; la comparación es arquitectónica, no exactamente de capacidad.

## Plan

- F11.0 WordPiece.
- F11.1 Unigram.
- F11.2 SentencePiece.
- F11.3 Byte-level/BLT + informe.

## Límites

- WordPiece se entrenó con 2k piezas, muy por debajo de BERT.
- La comparación con BPE es a igual presupuesto, no a igual calidad.
- El fallback de bytes cambia el WordPiece estándar (lossless a propósito).
