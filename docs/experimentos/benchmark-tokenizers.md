# Benchmark multidimensional de tokenizers para español

## Por qué

F9 midió compresión y BPB, pero no la hipótesis real: **¿una segmentación morfológica ayuda a que el
modelo interprete el idioma?** El paper de GEEToken-Bench muestra que la tokenización se evalúa mejor
en varias dimensiones y contra una referencia, no solo con tokens/carácter.

Este benchmark es para **español narrativo**, no para código GEE. Del paper se toma el método, no sus
datos ni sus números.

## Alcance

- Corpus principal: Quijote 70/15/15.
- Externo: Lazarillo.
- Referencia morfológica: una muestra de formas verbales anotada con tallo, terminación y tiempo
  (pasado/presente/futuro).
- Comparación controlada: mismo LM, mismos datos, mismos pasos; solo cambia el tokenizer.
- Nada de etiquetas semánticas que no se puedan reconstruir: todo sigue siendo lossless.

## F10.0 Corpus y referencia morfológica

1. Extraer formas verbales del Quijote y Lazarillo.
2. Construir una referencia con:
   - `forma` original;
   - `tallo`;
   - `sufijo`;
   - `tiempo` (`pasado`, `presente`, `futuro`, `ambiguo`);
   - `confianza` y nota de ambigüedad.
3. Fuente inicial: lexicón de terminaciones + revisión manual de una muestra.
4. Mínimo: 500 formas, 100 revisadas a mano; se reporta el acuerdo.
5. Si no hay analizador morfológico fiable, la referencia se declara **hipótesis**, no verdad.

### Estado F10.0

Referencia construida con `spacy` + `es_core_news_sm` y revisada por la persona:

- 800 formas finitas: 400 Quijote + 400 Lazarillo.
- 100 decisiones: **95 conservadas, 5 rechazadas** (adverbios, nombres y tokens mal etiquetados).
- **8 correcciones** aplicadas: `salgamos`, `remediaran`, `comenzamos`, `pegan`, `quedase`,
  `entendiera`, `ponérsele` y `Ponme`.
- Referencia final: **795 formas**; 95 revisadas; el resto queda como original no revisado.
- Archivos: `data/morphology-reference-reviewed.json`, `morphology-review-fixed.csv`,
  `morphology-review-decisions.json`, `morphology-review-audit.json`.

La auditoría no encuentra fallos en las 95 filas corregidas. Las irregulares sin tallo quedan con
`confidence=baja` y se excluyen de las métricas de frontera; siguen disponibles para la sonda de
tiempo. La referencia es una **hipótesis**, no una verdad lingüística.

## F10.1 Métricas y arnés

Comando: `node lab/benchmark/metrics.ts` → `data/tokenizer-benchmark.json`.

Resultados sobre Quijote test y referencia morfológica:

| Tokenizer | Vocab | Tokens/car. | Frontera F1 | ED | BLEU | BPB proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BPE 2048 | 2.048 | 0,284 | 0,540 | 0,567 | 0,309 | **3,068** |
| BPE + sufijos | 2.084 | 0,284 | **0,551** | **0,522** | **0,367** | 3,072 |
| BPE + prefijos + sufijos | 2.097 | 0,284 | **0,551** | **0,522** | **0,367** | 3,074 |
| Palabra | 2.048 | 0,525 | 0,232 | 0,957 | 0,008 | 4,701 |
| Carácter | 269 | 1,000 | 0,321 | 0,868 | 0,012 | 6,585 |

**Primera señal positiva:** los sufijos mejoran frontera (+0,011), ED (-0,045) y BLEU (+0,058) sin
cambiar tokens/carácter. Pero el BPB **empeora** 0,004. Es el trade-off que buscábamos: morfología
para estructura/interpretación, no para compresión.

Cuidado: la referencia favorece por diseño el corte tallo+sufijo. El BPB es la métrica independiente;
F10.2 debe confirmar si el modelo usa esa morfología.

| Dimensión | Métrica | Qué mide |
| --- | --- | --- |
| Exactitud | `decode(encode(x)) == x` | Cobertura lossless |
| Compresión | tokens/carácter, CR | Eficiencia |
| Fronteras | Precision/Recall/F1 morfológica | ¿Corta tallo/sufijo donde toca? |
| Secuencia | Edit Distance, BLEU | Parecido con la referencia |
| Diversidad | NUT, LD | Uso de vocabulario |
| Modelo | BPB | Calidad del LM |
| Semántica | accuracy de tiempo verbal | ¿El modelo distingue pasado/presente/futuro? |

El arnés:

- construye el mismo conjunto de tokens para todos los tokenizers;
- verifica exactitud en train/test/externo;
- guarda un JSON por tokenizer y un resumen.

## F10.2 Sonda de tiempo verbal

Comando: `python lab/benchmark/tense_probe.py`.

Se entrenó el mismo transformer (1,9M) para cada tokenizer, se congeló y se ajustó una regresión
logística sobre el estado oculto en la posición del verbo. Entrenamiento: 80% de las formas de
Quijote; test: 20% de Quijote + Lazarillo.

| Tokenizer | Accuracy Quijote | Accuracy externo | Baseline mayoría |
| --- | ---: | ---: | ---: |
| BPE 2048 | **0,913** | 0,773 | 0,506 |
| BPE + sufijos | 0,850 | **0,793** | 0,506 |
| Carácter | **0,950** | **0,866** | 0,506 |

La morfología **no gana de forma consistente**: mejora el externo +0,020 pero pierde Quijote -0,063.
Carácter gana ambas, probablemente porque 128 tokens cubren muchos menos caracteres y la sonda ve la
terminación completa; **no es una comparación de contexto equivalente**.

Conclusión provisional: la morfología no mejora la sonda; el resultado de carácter es una señal de
que la granularidad y el contexto efectivo importan más que el sufijo aislado. F10.3 debe ajustar el
contexto por bytes antes de concluir.

## F10.3 Comparación controlada e informe

### Evolución con nuevo fitness

Genoma nuevo:

- `modo`: ninguno, sufijo, prefijo+sufijo, prefijo+sufijo+número.
- `minLength`: 1–6.
- `presupuesto`: 512–2.048 tokens.
- `minSuffix`: 2–5.

Fitness nuevo:

```
score = tokens/carácter + 0.5 * (1 - F1_frontera) + 0.1 * ED
```

No incluye BPB; es una fitness de **estructura**, no de modelo.

| Individuo | Vocab | Tokens/car. | F1 | ED | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| BPE base | 2.048 | 0,284 | 0,540 | 0,567 | 0,571 |
| Ganador SGE/aleatorio | 2.097 | 0,284 | **0,551** | **0,522** | **0,560** |

- 672 combinaciones, 100×100 SGE; se agotan y aleatorio/SGE empatan.
- El nuevo fitness **sí selecciona morfología**: prefijo+sufijo, porque mejora frontera/ED al mismo
  coste de tokens.
- El prefijo no aporta sobre el sufijo solo; el desempate lo elige por vocabulario.
- No se puede concluir que sea mejor para el LM: BPB queda fuera de esta fitness.

Variantes:

- BPE 2048;
- BPE + sufijos;
- BPE + prefijos + sufijos;
- tokenizer “morfológico” explícito;
- un control sin morfología.

Reportar:

- tabla de métricas;
- frente de Pareto;
- diferencias con intervalos de confianza;
- límites: referencia discutible, corpus pequeño, LM pequeño.

## Relación con F9

F9 queda como evidencia de compresión y BPB. F10 no repite 100×100: hace una comparación controlada
y explicable. Si F10 tampoco encuentra señal, la hipótesis morfológica se cierra con mejor evidencia.

## Riesgos

- La referencia puede favorecer al tokenizer que corta como la referencia.
- El analizador puede confundir formas ambiguas (`era`, `fue`, `había`).
- Un LM pequeño puede no explotar la morfología.
- Añadir tokens puede mejorar una sonda y empeorar BPB; hay que reportar ambos.

## Criterio de cierre

- Referencia construida y documentada.
- Métricas reproducibles.
- Sonda de tiempo con resultado.
- Informe con recomendación: morfología sí/no, para qué.
