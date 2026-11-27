# Evolución de tokenizadores sin pérdida

**Estado:** F9.0/F9.1 en revisión; F9.2 (piloto) siguiente.
**Pregunta:** ¿puede la evolución gramatical encontrar un tokenizer sin pérdida que use menos tokens
que BPE 2048 en el Quijote, con el mismo presupuesto de vocabulario y generalizando a un texto externo?

## Estado F9.0/F9.1

Baselines con Quijote 70/15/15 y *Lazarillo de Tormes* externo:

| Tokenizer | Vocab | Val tokens/car. | Externo | Exacto |
| --- | ---: | ---: | ---: | --- |
| byte | 256 | 1,019 | 1,025 | sí |
| carácter (actual) | 76 | 1,000 | 1,000 | **no** |
| BPE 512 | 512 | 0,437 | 0,458 | sí |
| BPE 1024 | 1.024 | 0,344 | 0,365 | sí |
| **BPE 2048** | 2.048 | **0,285** | **0,313** | **sí** |
| palabra (actual) | 1.845 | 0,381 | 0,399 | **no** |

Los tokenizers de carácter y palabra actuales usan `<unk>`; no sirven para el experimento lossless.
F9.1 añade un tokenizer de bytes con fallback: siempre reconstruye, y los tokens compuestos solo
acortan.

| Programa | Vocab | Val tokens/car. | Externo | Exacto |
| --- | ---: | ---: | ---: | --- |
| palabra-512 + fallback | 916 | 0,599 | 0,638 | sí |
| carácter-512 + fallback | 269 | 1,000 | 1,000 | sí |

Conclusión parcial: **BPE 2048 es el baseline exacto a batir (0,285 / 0,313)**. El fallback lossless
de palabra/carácter cuesta tokens; la evolución tiene que encontrar una composición mejor, no solo
elegir familia.

## Piloto F9.2

- 20.000 caracteres de train, 8.000 de validación y 10.000 de *Lazarillo*.
- Genoma de 4 genes: familia `char/word/dict`, vocabulario 2.048/4.096, `maxLen` 32/64, frecuencia 1/2/5.
- 200 genotipos aleatorios + SGE 20×10; 36 combinaciones únicas (el espacio se agota).
- Mejor: `dict/4096/64/1`, **0,457** tokens/carácter val y **0,499** externo.
- Aleatorio y SGE empatan: con 36 combinaciones no hay margen para que la evolución demuestre nada.
- El piloto valida el arnés y la reparación; no es una comparación final contra BPE 2048 (otro train).

F9.3 necesita un genoma más rico: split, diccionario/frases, políticas y BPE; si no, la búsqueda se
reduce a enumerar.

## F9.3 primer barrido

- 20.000 caracteres de train, 8.000 de validación, 10.000 de *Lazarillo*.
- Genoma 5 genes: fuente `word/substring/char`, vocabulario 2k/4k/8k/16k/32k, `maxLen` 32/64/128,
  frecuencia 1/2/5/10 y política frecuencia/ganancia. **360 combinaciones**.
- 100×100 SGE con cache: 358 tokenizers únicos; aleatorio y SGE empatan porque el espacio se agota.
- Mejor: `substring/32768/32/1/frecuencia`, **0,429 val / 0,471 externo** (score 0,754).
- Por vocabulario: 2k 0,484; 4k 0,457; 8k 0,440; 16k 0,429; 32k 0,429.
- Por fuente: word 0,569; substring 0,429; char 1,000.

Conclusión: el diccionario de substrings domina a word/char, pero el **espacio sigue siendo pequeño**.
Para que 100×100 signifique algo hacen falta genes de composición (split, híbridos, BPE) y métodos
GE/πGE/HAEA, no solo un producto cartesiano de 360 configuraciones.

## Decisiones para F9.3a/F9.3b

Se amplía el genoma y se cambia el fitness:

### Contexto (pasado/futuro)

- `leftContext`: 0–3 caracteres/clases.
- `rightContext`: 0–3.
- `lookahead`: `2 caracteres` o `palabra completa`; **el genoma elige**.
- Segmentación con programación dinámica: se permite que el corte dependa del contexto izquierdo
  (pasado) y derecho (futuro); el decode sigue siendo exacto.

### Semántica temporal

- `tenseMode`: `none | suffix | typeTag`.
- `tenseLexicon`: terminaciones verbales (pasado/presente/futuro) o etiquetas de tipo.
- El tokenizer puede separar tallo + terminación (`habl-aba`, `habl-ará`) o insertar una etiqueta.
- **Cambia el fitness**: las etiquetas añaden tokens; si solo se minimizan tokens, la evolución las
  apaga. Se evalúan con **BPB de un LM proxy** además de tokens/carácter.

### Fitness multiobjetivo

Para F9.3a (solo contexto):

```
scoreTokens = tokensVal + 0.5 * tokensExternal + 0.1 * (1 - vocabUsage)
```

Para F9.3b (contexto + semántica), Pareto:

```
minimizar (tokensVal, BPB_proxy)
```

- Se guarda el frente de Pareto; no se colapsa a un solo número.
- El BPB se mide con un LM proxy pequeño; el LM final queda para F9.4.
- Un tokenizer con más tokens pero mejor BPB puede sobrevivir.

### Genoma v2

```
split, encoder, orden
vocabSize, maxTokenBytes, minFrequency, policy
leftContext, rightContext, lookahead (2 chars | palabra)
tenseMode, tenseLexicon
fallback
```

## Fases de F9.3

- **F9.3a**: contexto y lookahead; fitness de tokens.
- **F9.3b**: semántica temporal; Pareto tokens vs BPB proxy.
- **F9.3c**: 100×100 con genoma v2, GE/SGE/πGE/HAEA.
- **F9.4**: finalistas con LM completo y BPB real.
- **F9.5**: informe y artefactos.

## Decisiones cerradas

- **Precisión = reconstrucción exacta**: `decode(encode(texto)) === texto`.
- Vocabularios a probar: **2.048, 4.096, 8.192, 16.384 y 32.768**.
- Longitud máxima de token: **32, 64 y 128 bytes**.
- Corpus principal: Quijote 70/15/15.
- Texto externo: *Lazarillo de Tormes* (dominio público), solo para evaluar.
- Fitness inicial **barato**, sin LM: tokens/carácter en validación y externo.
- Al final, entrenar un LM con el mejor tokenizer de cada método y compararlo contra los mejores
  métodos existentes (byte, carácter, palabra, BPE).
- Población 100, generaciones 100, con fitness barato.
- GPU aceptada para el LM; tokenizer y evolución en TypeScript.

## Por qué Grammatical Evolution

El tokenizer no es un número: es un **programa** (normalizar, partir, construir vocabulario, fallback).
GE permite representar ese programa con una gramática; SGE y πGE reducen la redundancia y la dependencia
de la posición.

| Variante | Genotipo | Ventaja | Coste |
| --- | --- | --- | --- |
| GE | codones planos, expansión en profundidad | simple, histórica | redundancia, wrapping, posición |
| SGE | una lista de codones por no-terminal | localidad, cruce por no-terminal | genoma más estructurado |
| πGE | pares `(nont, rule)` + orden | independiente de la posición | genoma más grande |
| HAEA | codones + tasas adaptativas | explota buenos operadores | más parámetros |
| Grammatical Swarm | PSO sobre genomas | diversidad | menos explored en este dominio |
| Memético | evolución + búsqueda local | afina vocab/maxLen | presupuesto extra |
| RL | política de operadores/expansión | adaptativo | muestras caras; fase tardía |

## Genoma

```
Tokenizer  → Segment* Fallback
Segment    → Split(clase) Encoder
Encoder    → Byte | Char | Word | BPE(policy) | Dict(maxLen)
           | GreedyDict | DP(coste) | Híbrido
Vocabulary → Frecuencia | PMI | Ganancia | Sufijos | Frases
```

Genes:

- split: nada, palabras/números/signos/espacios, clases Unicode, regex DSL;
- encoder por clase;
- política de vocabulario y longitud;
- `vocabSize`, `maxTokenBytes`, `minFrequency`;
- fallback byte/carácter.

No se guardan los merges en el genoma: se derivan del corpus con la política elegida. Para 32.768 se
precomputan candidatos/merges una vez y el genoma selecciona; si no, 100×100 no es viable.

## Reparación y cobertura

Después de decodificar el genoma:

1. construir el tokenizer;
2. comprobar `decode(encode(Quijote)) === Quijote`;
3. comprobar el externo;
4. si falta cobertura, añadir bytes/caracteres, activar fallback y penalizar;
5. descartar si sigue sin ser lossless.

## Fitness

```
lossless = decode(encode(train)) == train && decode(encode(val)) == val
tokensVal = len(encode(val)) / len(val)
tokensExternal = len(encode(external)) / len(external)
vocabUsed = tokens usados / vocabSize
penalty = 0.1 * (1 - vocabUsed) + 0.05 * rareTokens + 0.05 * oversizedTokens
score = tokensVal + 0.5 * tokensExternal + penalty
```

- Objetivo principal: minimizar `tokensVal`.
- Desempate: externo, vocabulario usado, velocidad.
- `BPB` se reserva para la fase con LM.
- Se guarda el frente de Pareto: tokens vs vocabulario vs velocidad.

## Búsqueda

- **F9.0**: corpus, checker, baselines.
- **F9.1**: gramática, genoma, decodificador, reparación, diccionario/BPE.
- **F9.2**: piloto 20×10 con 2.048/4.096 y `maxLen` 32/64.
- **F9.3**: 100×100 para los cinco vocabularios y tres `maxLen`; GE/SGE/πGE/HAEA.
- **F9.4**: mejor de cada método + baselines, LM 1,4M en GPU, BPB en Quijote test y externo.
- **F9.5**: informe y artefactos.

## Artefactos

```
artifacts/tokenizer-evolution/<id>/
  genome.json
  tokenizer.json
  metrics.json
  grammar.json
```

En F9.4 se añade `model.bin` y `bpb.json`. Top-k por Pareto; no se versionan los 10.000.

## Riesgos

- Sobreajuste al Quijote: holdout y externo obligatorios.
- Vocabulario muerto: penalización y frecuencia mínima.
- Tokenizers degenerados: `maxTokenBytes` y tokens usados.
- BPE 32k: precomputar o usar diccionario.
- Coste de evaluación: cachear por hash de genoma y deduplicar fenotipos.
- Confundir tokens con calidad: BPB solo con LM.

## Relación con la arquitectura

La arquitectura se evolucionará después, no a la vez: genes de bloques, `d_model`, cabezas, `d_ff`,
activación, norm, atención, contexto; restricciones de divisibilidad y presupuesto; fitness BPB con el
tokenizer ganador. Hacerlo conjunto confunde la causa.

## F9.3a contexto: resultado negativo

- Vocabulario 4.096 (bytes + substrings + frases de hasta 4 unidades), `maxLen` 64.
- 20.000 train / 8.000 validación / 10.000 externo.
- Greedy byte: **0,279 val / 0,326 externo**.
- DP sin contexto: **0,276 / 0,324**.
- DP con contexto 2/2, 3/3 palabra, 1/1 fuerte y asimetrías pasado/futuro: **0,276 / 0,324**,
  exactamente igual.

**Por qué no cambia nada:** con un vocabulario fijo y coste 1 por token, el DP ya encuentra la
segmentación de mínimo número de tokens. El contexto solo puede **desanimar** cortes; nunca reduce el
número de tokens, y el lookahead no añade tokens nuevos. Para que el contexto importe hay que usarlo
para **elegir el vocabulario** o para **BPB**, no como penalización de un DP que ya es óptimo.

Conclusión: F9.3a valida el tokenizer contextual y su pérdida cero, pero no mejora tokens. La semántica
y el contexto solo tienen sentido en F9.3b con **BPB** y multiobjetivo.

## F9.3b semántica temporal: resultado negativo

Proxy: transformer de 32 dimensiones, 1 bloque, 300 pasos, contexto 64; BPB sobre validación y externo.

| Tokenizer | Vocab | Val tokens/car. | Externo | BPB val | BPB externo | Exacto |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Base (sin semántica) | 4.043 | **0,279** | **0,326** | **3,263** | **3,812** | sí |
| Terminaciones verbales | 4.087 | 0,484 | 0,521 | 4,179 | 5,090 | sí |
| Etiquetas de tipo | 4.047 | 0,558 | 0,653 | 4,807 | 5,574 | sí |

La semántica **no paga** en este presupuesto: separar `habl-aba` fragmenta palabras y las etiquetas
duplican tokens. El LM proxy no compensa; el frente de Pareto lo domina el tokenizer base. La idea de
tiempo verbal queda registrada, pero para que aporte haría falta un LM más grande, más datos o una
tarea de morfología, no solo cambiar el tokenizer.

## F9.3c pendiente

Barrido 100×100 con el genoma v2 completo y GE/SGE/πGE/HAEA. Antes hay que decidir si la semántica
entra como gen apagado por defecto o se reserva para una fase con LM real.

## F9.3c primer barrido v2

- 480 combinaciones: 4 fuentes (word/substring/both/char), vocab 2k–32k, `maxLen` 32/64/128,
  frecuencia 1/2/5/10 y política frecuencia/ganancia.
- 100×100 por método (SGE, GE, πGE, HAEA); 480 tokenizers únicos; todas las combinaciones evaluadas.
- **Todos los métodos empatan**: `0,429` val / `0,471` externo.
- Mejor por vocabulario: 2k 0,474; 4k 0,450; 8k 0,433; 16k 0,429; 32k 0,429.
- El vocabulario satura en **16.384**; 32.768 no mejora. La fuente `substring`/`both` domina a
  `word`/`char`.

Otra vez el espacio se agota: 480 combinaciones no permiten que GE/SGE/πGE/HAEA demuestren ventaja.
Para F9.3c “de verdad” hacen falta genes de composición (split, híbridos, BPE) que multipliquen el
espacio, no solo el producto cartesiano actual.

## F9.4 primer proxy con LM

La corrida de 1M en CPU era demasiado lenta; este proxy usa `d_model=32`, 1 bloque, 200 pasos y
contexto 64 sobre 20k/8k/8k de Quijote y 10k de Lazarillo.

| Tokenizer | Vocab | BPB test | BPB externo |
| --- | ---: | ---: | ---: |
| BPE 2048 (entrenado en 20k) | 1.584 | **3,280** | **3,787** |
| substring-4k | 4.096 | 4,518 | 5,161 |
| palabra | 628 | 4,930 | 5,739 |
| carácter | 267 | 6,586 | 7,232 |

**BPE gana en BPB**, aunque substring-4k tenga menos tokens/carácter en el barrido. Otra vez:
menos tokens no equivale a mejor modelo. El tokenizer evolucionado no supera al baseline en este
presupuesto; la corrida de 1M queda pendiente de GPU.

## F9.4b GPU y morfología verbal

La siguiente pregunta: **¿mejora el LM si el tokenizer separa morfología verbal sin aumentar tokens?**

Diseño:

1. **Base**: BPE 2048 (el mejor BPB del proxy) y el mejor substring.
2. **Sufijos**: lexicón de terminaciones de pasado/presente/futuro.
3. **Post-BPE**: si una palabra termina en un sufijo y BPE la parte en varias piezas, se reemplazan
   las piezas finales por **un token de sufijo**. Solo se acepta si el número de tokens no aumenta.
4. **Variantes**:
   - BPE solo;
   - BPE + sufijos (sin etiquetas);
   - BPE + etiqueta de tiempo (una etiqueta por verbo, sabiendo que añade tokens);
   - substring + sufijos;
   - BPE + números/fechas como tokens propios.
5. **Evaluación**: LM 1.4M en GPU, mismo presupuesto, BPB en Quijote test y Lazarillo.

Otras ideas que merecen una ablación, no todas a la vez:

- números, fechas y horas como tokens;
- mayúsculas/forma de palabra;
- prefijos (`des-`, `re-`, `in-`) además de sufijos;
- BPE dropout / segmentaciones alternativas;
- tokens de código/URL para un corpus técnico.

La regla: **cada idea debe reducir o mantener tokens y mejorar BPB**. Si solo añade tokens, se queda
como ablación negativa.

## F9.4b morfología verbal: proxy y GPU

BPE 2048 frente a BPE + sufijos (post-BPE, solo si los tokens no aumentan).

Proxy de 32 dimensiones y 300 pasos:

| Tokenizer | Vocab | Train tokens | Test tokens | Externo | BPB test | BPB externo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | 36.214 | 8.417 | 30.749 | **3,023** | **3,301** |
| BPE + sufijos | 2.084 | 36.183 | 8.405 | 30.703 | 3,032 | 3,310 |

GPU: transformer `d_model=160`, 4 bloques, contexto 128, batch 16, 400 pasos, AdamW.

| Tokenizer | Vocab | BPB test | BPB externo |
| --- | ---: | ---: | ---: |
| BPE 2048 | 2.048 | 2,4006 | **2,6752** |
| BPE + número | 2.048 | 2,4006 | **2,6752** |
| BPE + prefijo | 2.061 | 2,3988 | 2,6962 |
| BPE + sufijo | 2.084 | 2,3999 | 2,6932 |
| BPE + prefijo + sufijo | 2.097 | 2,4027 | 2,6938 |
| BPE + prefijo + sufijo + número | 2.097 | 2,4027 | 2,6938 |
| BPE + mayúsculas | 2.107 | **2,3840** | 2,6833 |
| BPE + forma | 2.050 | 2,4060 | 2,6783 |
| BPE + dropout 10% | 2.048 | 2,5044 | 2,7723 |

En GPU todas las variantes quedan entre **2,38 y 2,50 test** y **2,67–2,77 externo**. Mayúsculas gana
el test por 0,017 pero pierde 0,008 en externo; dropout y forma de palabra empeoran. Ninguna
combinación supera el externo de BPE. La morfología y la forma no pagan con este tokenizer ni con este
presupuesto; quedan como ablaciones negativas.

## GPU

El evaluador está en `lab/tokenizers/train_gpu.py`; exporta tokens desde TS y entrena en PyTorch/CUDA.
La instalación de `torch+cu126` se completó y la GTX 1060 3GB entrena el modelo de ~1,9M.

Próximas variantes que sí merecen prueba: forma de palabra, BPE dropout, tokens de código/URL.
Cada una debe mantener o reducir tokens y mejorar BPB.
