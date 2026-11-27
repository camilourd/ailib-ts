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
