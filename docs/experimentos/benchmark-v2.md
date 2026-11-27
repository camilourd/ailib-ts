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

## F12.2d BPE con prefijo+sufijo

Misma receta, 100 pasos, 3 semillas. Se añadieron prefijos y sufijos españoles al vocabulario BPE 2k.

| Dominio | Tokenizer | BPB test | BPB validación |
| --- | --- | ---: | ---: |
| Narrativo | **BPE 2k puro** | **2,854 ± 0,005** | **2,832 ± 0,007** |
| Narrativo | BPE + prefijo + sufijo | 2,857 ± 0,008 | 2,835 ± 0,011 |
| Técnico | **BPE 2k puro** | **3,948 ± 0,023** | **3,090 ± 0,005** |
| Técnico | BPE + prefijo + sufijo | 3,952 ± 0,026 | 3,094 ± 0,013 |
| Código | BPE 2k puro | 3,937 ± 0,019 | 3,820 ± 0,013 |
| Código | **BPE + prefijo + sufijo** | **3,920 ± 0,032** | **3,810 ± 0,036** |

- En **narrativo y técnico el híbrido no ayuda**; queda igual o un poco peor.
- En **código** mejora 0,018 en test y 0,010 en validación, pero con más varianza.
- La ganancia de F11 en Quijote **no se replica** en estos corpus.
- El híbrido apenas reduce tokens: 9.016→9.007 en narrativo, 14.009→13.995 en técnico,
  17.518→17.488 en código.

## F12.2e Modern con parámetros más cerca

La primera idea para mejorar Modern: bajar `d_ff` de SwiGLU de 1.024 a **832**, acercando parámetros
al transformer clásico. Narrativo, BPE 2k, 100 pasos, 3 semillas:

| Modelo | Params | BPB test | BPB validación |
| --- | ---: | ---: | ---: |
| Clásico ReLU | 5.789.696 | 2,854 ± 0,005 | 2,832 ± 0,007 |
| **Modern SwiGLU d_ff=832** | 6.478.336 | **2,825 ± 0,011** | **2,803 ± 0,008** |

- **Modern gana por primera vez de forma consistente en F12**: 0,029 en test y 0,029 en validación.
- No es una comparación de parámetros exacta: Modern sigue teniendo **6,48M vs 5,79M**.
- Es solo narrativo y 100 pasos; falta técnico, código y 400 pasos.

## F12.3 retrieval: intento descartado

Se implementó retrieval de frase siguiente con BM25. En narrativo:

| Tokenizer | R@1 | R@5 |
| --- | ---: | ---: |
| BPE 2k | 0,000 | 0,161 |
| BPE + prefijo + sufijo | 0,000 | 0,152 |
| BPE 8k | 0,000 | 0,125 |

La tarea es demasiado difícil: frases consecutivas no comparten suficientes términos. **No discrimina
tokenizers** y se descartó antes de correr técnico/código.

## F12.2f weight decay

Modern d_ff=832, narrativo, BPE 2k, 100 pasos, semilla 1:

| Configuración | BPB test | BPB validación |
| --- | ---: | ---: |
| Sin weight decay | **2,833** | **2,805** |
| Weight decay 0.1 | 2,833 | 2,805 |

No cambia nada a 100 pasos. El weight decay se implementó igualmente en `src/optim/sgd.ts` con su
prueba.

## F12.2g dropout, tying y LayerScale

Modern d_ff=832, narrativo, BPE 2k, 100 pasos, **semilla 1**:

| Variante | Params | Test | Validación |
| --- | ---: | ---: | ---: |
| Modern d_ff=832 | 6.478.336 | 2,833 | 2,805 |
| + dropout 0.1 | 6.478.336 | **2,828** | **2,802** |
| + LayerScale | 6.481.408 | 2,829 | 2,802 |
| + tied embeddings | 5.952.000 | 2,928 | 2,897 |

- Dropout y LayerScale mejoran **0,004–0,005**, dentro del ruido de una semilla.
- **Tied embeddings empeora claramente** (2,928/2,897), aunque baja a 5,95M parámetros.
- Las tres piezas quedaron en `src/nn` y `src/models` con tests; el `.py` usa las mismas opciones.
- Ninguna de estas variantes supera todavía la media de 3 semillas de Modern d_ff=832
  (2,825/2,803).

## F12.4 variantes de BPE

### Vocabulario

Tokens/carácter con BPE por dominio, 50k de train:

| Dominio | 1k | 1,5k | 2k | 3k | 4k |
| --- | ---: | ---: | ---: | ---: | ---: |
| Narrativo | 0,354 | 0,317 | **0,301** | 0,285 (2.930) | 0,285 (2.930) |
| Técnico | 0,501 | 0,460 | 0,430 | 0,403 | **0,391 (3.745)** |
| Código | 0,442 | 0,407 | 0,395 | 0,385 (2.864) | 0,385 (2.864) |

- 2k es mejor que 1k/1,5k.
- 3k/4k solo comprimen más cuando el vocabulario real todavía crece; en narrativo/código ya está
  topado y no cambia nada.
- No hay evidencia de que 3k/4k mejoren el BPB: en F12.2, 8k ya empeoró.

### Mezcla de dominios

BPE 2k entrenado con 50k de cada dominio, evaluado en cada uno:

| Dominio | BPE propio | BPE mezcla |
| --- | ---: | ---: |
| Narrativo | **0,301** | 0,349 |
| Técnico | **0,430** | 0,444 |
| Código | 0,395 | **0,368** |

La mezcla ayuda a código y perjudica narrativo/técnico.

### BPE-dropout

BPE 2k con dropout de merges 0.1, en test:

| Dominio | Sin dropout | Con dropout |
| --- | ---: | ---: |
| Narrativo | 0,302 | 0,412 |
| Técnico | 0,430 | 0,518 |
| Código | 0,397 | 0,511 |

Aumenta tokens/carácter; no es una mejora de compresión. Queda implementado en `src/data/bpe-dropout.ts`
con test, pero no hay evidencia para usarlo en el LM.

## F12.5 pre-tokenización regex y preposiciones

### BPE con regex estilo GPT-2

Tokens/carácter a 2k:

| Dominio | BPE normal | BPE regex |
| --- | ---: | ---: |
| Narrativo | **0,301** | 0,589 |
| Técnico | **0,430** | 0,667 |
| Código | **0,395** | 0,651 |

La pre-tokenización **casi duplica los tokens**: al no permitir merges entre palabras, pierde la
compresión que da BPE. No se llevó al LM; el coste en longitud es demasiado grande.

### BPE + prefijos + sufijos + preposiciones

Seed 1, 100 pasos, vocabulario 2k + extras:

| Dominio | Híbrido pref/suf | + preposiciones |
| --- | ---: | ---: |
| Narrativo | **2,848 / 2,824** | 2,850 / 2,833 |
| Técnico | **3,939 / 3,093** | 3,990 / 3,096 |
| Código | **3,931 / 3,837** | 3,938 / 3,834 |

- Narrativo y técnico empeoran.
- Código mejora 0,003 en validación pero empeora 0,007 en test.
- No hay señal para correr 3 semillas. Las preposiciones no ayudan.

## F12.6 Modern 832 en todo el benchmark

3 dominios × BPE 2k/8k × 3 semillas, 100 pasos:

| Dominio | Tokenizer | Clásico test/val | **Modern 832 test/val** |
| --- | --- | ---: | ---: |
| Narrativo | BPE 2k | 2,854 / 2,832 | **2,825 / 2,803** |
| Narrativo | BPE 8k | 2,902 / 2,877 | **2,897 / 2,868** |
| Técnico | BPE 2k | 3,948 / 3,090 | **3,871 / 3,026** |
| Técnico | BPE 8k | **4,065 / 3,060** | 4,074 / 3,056 |
| Código | BPE 2k | **3,937 / 3,820** | 3,955 / 3,824 |
| Código | BPE 8k | **4,152 / 3,994** | 4,201 / 4,023 |

- **Modern gana en narrativo (2k y 8k) y técnico con 2k.**
- Pierde en técnico con 8k por 0,009 y en código (2k y 8k).
- **BPE 2k es el mejor tokenizer para Modern en los tres dominios**; 8k nunca gana.
- Modern usa 6,48M–7,35M según vocab, frente a 5,79M–6,66M del clásico.
- El mejor Modern del benchmark es **narrativo + BPE 2k: 2,825 / 2,803**.
- No alcanza al mejor global **Unigram + clásico 400: 2,631 / 2,814**, pero ese es otro presupuesto.

## F13.1 primer MoE

MoE de 4 expertos, top-1, FFN ReLU, narrativo, BPE 2k, 100 pasos, semilla 1:

| Modelo | Params totales | BPB test | BPB validación |
| --- | ---: | ---: | ---: |
| Clásico denso | 5.789.696 | 2,854 | 2,832 |
| Modern 832 | 6.478.336 | 2,825 | 2,803 |
| **MoE 4× top-1** | **15.256.088** | **2,827** | **2,802** |

- No mejora al Modern: queda **2,827/2,802** con **2,6× más parámetros totales**.
- El uso por experto está **desbalanceado**: en la última capa hay expertos con 109.207 tokens y otros
  con 2.117. El router colapsa parcialmente.
- Sin `load-balancing loss`, el resultado no es concluyente sobre el routing.
- Params activos ≈ un experto + router; total incluye 4 expertos.

## F13.0 MoE en mezcla de dominios

Train mezclado 50k de cada dominio, BPE 2k, 100 pasos, semilla 1:

| Modelo | Params | Narrativo | Técnico | Código | Media |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clásico denso | 5.789.696 | 2,991 | 4,220 | 3,986 | 3,733 |
| MoE 4× top-1 | 15.256.088 | **2,977** | **4,173** | **3,982** | **3,711** |
| MoE + balance 0.01 | 15.256.088 | 2,976 | 4,173 | 3,982 | 3,710 |
| MoE router aleatorio | 15.256.088 | 2,984 | 4,180 | 3,987 | 3,717 |

- El **balanceo funciona**: el uso por experto pasa de estar desbalanceado a ~25% cada uno.
- Pero **no mejora el BPB**: 2,976/2,947 frente a 2,977/2,946 sin balanceo.
- El **router aleatorio** queda peor (2,984/2,952) que el aprendido (2,977/2,946), pero por poco.
- El router aprendido no se especializa por dominio: las distribuciones de uso son casi iguales.
- La mejora sobre el denso es pequeña y viene acompañada de **2,6× parámetros totales**.

## F14: 15M en la GTX 1060

Configuración:

| Campo | 5,8M | 15M |
| --- | --- | --- |
| `d_model` | 256 | **384** |
| Bloques | 6 | **8** |
| `d_ff` | 1.024 | **1.536** |
| Cabezas | 8 | 8 |
| Params | 5.789.696 | **15.771.392** |
| Batch | 2 | 2 |
| Grad accum | 4 | 4 |
| Activation checkpoint | no | **sí** |
| Contexto | 128 | 128 |

3 semillas, 100 pasos, train completo de los 3 dominios (133.881 tokens):

| Modelo | Test | Validación | VRAM pico asignada | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| 5,8M | 2,941 ± 0,013 | 2,926 ± 0,012 | 136 MB | 9.018 |
| **15M** | **2,893 ± 0,019** | **2,881 ± 0,014** | **329 MB** | 4.399 |

Por dominio:

| Dominio | 5,8M | 15M |
| --- | ---: | ---: |
| Narrativo | 2,941 | **2,893** |
| Técnico | 4,108 | **4,004** |
| Código | 3,117 | **2,772** |

- **15M cabe en 3 GB** con activation checkpointing, batch 2 y acumulación 4: 329 MB de tensores
  asignados; el proceso completo usa ~2,4 GB por el contexto CUDA.
- Mejora **0,049 de test** y **0,045 de validación** frente al 5,8M.
- La mejora grande está en **código** (0,345); narrativo 0,049 y técnico 0,103.
- Cuesta **~2× tokens/s** y ~2,4× memoria de tensores.
- No se usó shared GPU memory.

## F14.2 crecimiento 6 → 8 capas

6 capas (`d_model=384`, `d_ff=1536`) durante 50 pasos, luego se insertan 2 capas con
**zero-init identity**, y 50 pasos más. Mismos datos y batch que F14.

| Modelo | Params | Test | Validación | Código |
| --- | ---: | ---: | ---: | ---: |
| 8 capas desde cero | 15.771.392 | **2,879** | **2,877** | **2,749** |
| **Progresivo 6→8** | 15.771.392 | 2,913 | 2,905 | 3,024 |

- La inserción **preserva la función exactamente**: `max|logits_antes − logits_después| = 0,0`.
- El modelo de 6 capas tras 50 pasos: 2,975 narrativo, 4,199 técnico, 3,504 código.
- Tras crecer y 50 pasos más: 2,913 / 4,137 / 3,024.
- **No alcanza al de 8 capas desde cero**, sobre todo en código.
- Mismos parámetros y mismos 100 pasos totales, pero las capas viejas solo vieron 50 pasos y las
  nuevas otros 50. El crecimiento identity funciona; el presupuesto por etapa no alcanza.

## Siguiente

- Repetir **BPE 2k vs BPE 8k** a 400 pasos y 3 semillas en el dominio donde importe.
- No gastar GPU en WordPiece/Unigram con este corpus.
- Añadir una tarea downstream por dominio.
