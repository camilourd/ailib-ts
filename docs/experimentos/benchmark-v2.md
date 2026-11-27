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

## Siguiente

- Pasar **BPE y SentencePiece 8k** a F12.2 con LM 5,8M.
- Probar WordPiece/Unigram solo si se entrenan con más datos o vocabulario menor.
- Añadir una tarea downstream por dominio.
