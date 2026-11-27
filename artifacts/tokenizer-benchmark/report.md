# Benchmark multidimensional de tokenizers para español

**Estado:** cerrado · **Repositorio:** ailib-ts · **Fecha:** 2026-11-27

## Pregunta

¿La morfología verbal en el tokenizer ayuda al modelo a interpretar el tiempo y a representar el
idioma, o solo cambia la compresión?

## Qué se construyó

- Referencia morfológica de **795 formas verbales finitas** del Quijote y Lazarillo.
- **100 decisiones de revisión**: 95 conservadas, 5 rechazadas, 8 correcciones.
- Arnés de métricas: exactitud, tokens/carácter, CR, frontera, ED, BLEU, NUT/LD y BPB proxy.
- Sonda de tiempo con LM congelado y regresión logística.
- Evolución con genoma morfológico y fitness multidimensional.

## F10.0: referencia

| Dato | Valor |
| --- | ---: |
| Formas totales | 795 |
| Decisiones revisadas | 100 |
| Conservadas | 95 |
| Rechazadas | 5 |
| Correcciones | 8 |
| Confianza alta | 656 |

La referencia es una **hipótesis**: spaCy puede etiquetar adverbios como verbos y las irregulares no
tienen tallo fiable. Se audita con `audit_fixed.py`; las formas de confianza baja se excluyen de
fronteras.

## F10.1: métricas

| Tokenizer | Vocab | Tokens/car. | Frontera F1 | ED | BLEU | BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | 0,284 | 0,540 | 0,567 | 0,309 | **3,068** |
| BPE + sufijos | 2.084 | 0,284 | **0,551** | **0,522** | **0,367** | 3,072 |
| BPE + prefijos + sufijos | 2.097 | 0,284 | **0,551** | **0,522** | **0,367** | 3,074 |
| Palabra | 2.048 | 0,525 | 0,232 | 0,957 | 0,008 | 4,701 |
| Carácter | 269 | 1,000 | 0,321 | 0,868 | 0,012 | 6,585 |

**Lectura:** los sufijos mejoran la estructura (frontera, ED, BLEU) sin coste de tokens. El BPB
empeora 0,004. Es un trade-off estructural, no una mejora de compresión.

## F10.2: sonda de tiempo

Mismo transformer (1,9M), congelado; regresión logística sobre el estado oculto en el verbo.
Entrenamiento 80% Quijote; test 20% Quijote + Lazarillo.

| Tokenizer | Quijote | Externo | Baseline mayoría |
| --- | ---: | ---: | ---: |
| BPE 2048 | **0,850** | **0,780** | 0,506 |
| BPE + sufijos | 0,800 | 0,763 | 0,506 |
| Carácter | **0,925** | **0,879** | 0,506 |

**Lectura:** la morfología **pierde** en la sonda. Carácter gana, pero 128 tokens de carácter cubren
muchos menos caracteres que 128 de BPE; **no es contexto equivalente**. El resultado no prueba que
carácter sea mejor, sino que la granularidad y el contexto efectivo pesan.

La sonda es determinista con semilla fija; dos corridas dieron los mismos valores.

## F10.3: evolución

Genoma: modo morfológico, longitud mínima, presupuesto y longitud mínima de sufijo.
Fitness:

```
tokens/carácter + 0,5·(1 - F1_frontera) + 0,1·ED
```

| Individuo | Vocab | Tokens/car. | F1 | ED | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPE base | 2.048 | 0,284 | 0,540 | 0,567 | 0,571 |
| SGE/aleatorio | 2.097 | 0,284 | **0,551** | **0,522** | **0,560** |

- 672 combinaciones; SGE y aleatorio empatan porque el espacio se agota.
- El nuevo fitness **selecciona prefijo+sufijo**.
- El prefijo no aporta sobre el sufijo solo.
- **No incluye BPB**: no prueba que sea mejor para el LM.

## Decisión

1. **No adoptar morfología como tokenizer por defecto** para el LM: no mejora BPB ni la sonda.
2. Mantenerla como **modo de interpretabilidad/estructura**, si se necesita segmentar morfología.
3. El baseline **BPE 2048** sigue siendo la referencia práctica.
4. Para cerrar la duda, el siguiente experimento debe:
   - igualar el **contexto por bytes**, no por tokens;
   - usar un LM más grande;
   - separar la referencia de la métrica de frontera;
   - incluir una tarea de tiempo verbal explícita.

## Reproducir

```bash
python lab/benchmark/build_reference.py
python lab/benchmark/audit_review.py
python lab/benchmark/repair_review.py
python lab/benchmark/audit_fixed.py
node lab/benchmark/metrics.ts
node lab/benchmark/export_probe.ts
python lab/benchmark/tense_probe.py
node lab/benchmark/evolve.ts
python lab/benchmark/check_artifacts.py
```

## Artefactos

- `data/morphology-reference-reviewed.json`
- `data/morphology-review-fixed.csv`
- `data/morphology-review-decisions.json`
- `data/tokenizer-benchmark.json`
- `data/tense-probe.json`
- `data/tokenizer-evolution-f10.json`
