# Benchmark v2: dominios y tokenizers

## F12.0 corpus

Tres dominios, split 70/15/15 por caracteres:

| Dominio | Fuente | Caracteres |
| --- | --- | ---: |
| Narrativo | Quijote | 200.001 |
| Técnico | docs `.md` | 217.051 |
| Código | `src` + `test` `.ts` | 295.348 |

El tokenizer se entrena **solo con train**; validación y test quedan fuera.
Archivos en `data/corpora/`.

## F12.1 zoo

Para cada dominio, con **50.000 caracteres comunes** de train:

- BPE, SentencePiece-BPE, WordPiece y Unigram.
- Vocabulario objetivo 2k/8k (16k se omitió: BPE/WordPiece ya mostraron su techo y era inviable).
- Métricas: exactitud, vocabulario real, tokens/carácter y, en narrativo, frontera/ED.
- Sin LM: primero decidir qué tokenizers merecen GPU.

Comando: `node lab/benchmark/tokenizer_zoo.ts`.

### Tokens/carácter (vocabulario real)

| Dominio | Tokenizer | 2k | 8k |
| --- | --- | ---: | ---: |
| Narrativo | BPE | 0,301 (2.048) | **0,285 (2.930)** |
| Narrativo | SentencePiece | 0,302 (2.048) | 0,287 (2.932) |
| Narrativo | WordPiece | 0,518 (1.604) | 0,518 (1.604) |
| Narrativo | Unigram | 0,691 (2.304) | 0,556 (8.448) |
| Técnico | BPE | **0,430 (2.044)** | **0,391 (3.745)** |
| Técnico | SentencePiece | 0,430 (2.048) | 0,391 (3.752) |
| Técnico | WordPiece | 0,640 (2.071) | 0,640 (2.071) |
| Técnico | Unigram | 0,756 (2.304) | 0,585 (8.448) |
| Código | BPE | **0,395 (2.048)** | **0,385 (2.864)** |
| Código | SentencePiece | 0,397 (2.048) | 0,386 (2.866) |
| Código | WordPiece | 0,558 (1.478) | 0,558 (1.478) |
| Código | Unigram | 0,586 (2.304) | 0,531 (6.336) |

### Frontera narrativa

| Tokenizer | F1 2k | F1 8k |
| --- | ---: | ---: |
| BPE | 0,537 | **0,550** |
| SentencePiece | 0,504 | 0,506 |
| WordPiece | 0,435 | 0,435 |
| Unigram | 0,361 | 0,363 |

## Lectura

- **BPE y SentencePiece dominan en los tres dominios**; BPE gana por poco.
- **WordPiece y Unigram no alcanzan el vocabulario objetivo** en 50k y quedan muy por detrás.
- El ranking **no cambia por dominio**: código y técnico se comportan como narrativo.
- En narrativo, BPE también gana frontera/ED.
- 8k mejora a BPE/SP en todos los dominios, pero el vocabulario real queda en 2,8k–3,7k.
- WordPiece/Unigram no justifican una corrida GPU con este corpus de 50k.

## Límites

- Técnico y código son del propio repo; no son un corpus externo independiente.
- 50k comunes, no el train completo.
- 16k omitido por techo de vocabulario y coste.
- Una semilla; el zoo no mide BPB ni tarea downstream.

## F12.2 cribado del LM

5,8M, 100 pasos, contexto 128, semilla 1, train de 50k. `validación` es el 15% reservado, no un
externo independiente.

| Dominio | Tokenizer | BPB test | BPB validación |
| --- | --- | ---: | ---: |
| Narrativo | **BPE 2k** | **2,853** | **2,831** |
| Narrativo | BPE 8k | 2,899 | 2,873 |
| Narrativo | SentencePiece 8k | 2,911 | 2,876 |
| Técnico | **BPE 2k** | **3,965** | 3,091 |
| Técnico | BPE 8k | 4,075 | **3,067** |
| Técnico | SentencePiece 8k | 4,100 | 3,108 |
| Código | **BPE 2k** | **3,952** | **3,820** |
| Código | BPE 8k | 4,150 | 3,992 |
| Código | SentencePiece 8k | 4,167 | 4,007 |

- **BPE 2k gana el test en los tres dominios.**
- Más vocabulario **no ayuda a BPB a 100 pasos**, aunque comprima más tokens.
- En técnico, BPE 8k gana por poco en validación (3,067 vs 3,091), pero pierde claro en test.
- SentencePiece 8k queda por detrás de BPE 8k en los tres dominios.
- El ranking de tokenizers **no cambia por dominio**: BPE 2k es el más robusto.

## F12.2b 400 pasos: señal negativa

Antes de gastar 6 corridas × 3 semillas, se probó **BPE 2k en narrativo a 400 pasos**:

| Tokenizer | 100 pasos | 400 pasos |
| --- | ---: | ---: |
| BPE 2k test | **2,853** | 2,909 |
| BPE 2k validación | **2,831** | 2,877 |

A 400 pasos **empeora**. Se detuvo la matriz de 400: repetir 3 semillas a 400 solo confirmaría el
sobreajuste. Las 3 semillas se hicieron a **100 pasos**, que es el presupuesto sano.

## F12.2c 3 semillas a 100 pasos

Media ± desviación estándar:

| Dominio | Tokenizer | BPB test | BPB validación |
| --- | --- | ---: | ---: |
| Narrativo | **BPE 2k** | **2,854 ± 0,005** | **2,832 ± 0,007** |
| Narrativo | BPE 8k | 2,902 ± 0,007 | 2,877 ± 0,009 |
| Técnico | **BPE 2k** | **3,948 ± 0,023** | 3,090 ± 0,005 |
| Técnico | BPE 8k | 4,065 ± 0,012 | **3,060 ± 0,007** |

- En narrativo, **BPE 2k gana test y validación**, con diferencia clara.
- En técnico hay **tradeoff**: BPE 2k gana test (3,948 vs 4,065), BPE 8k gana validación
  (3,060 vs 3,090), por 0,03.
- BPE 2k es más variable en test técnico (±0,023), pero gana la media.
- No hay evidencia para pagar el vocabulario 8k: comprime más, pero no mejora test en ningún dominio.

## Siguiente

- Repetir **BPE 2k vs BPE 8k** a 400 pasos y 3 semillas en el dominio donde importe.
- No gastar GPU en WordPiece/Unigram con este corpus.
- Añadir una tarea downstream por dominio.
