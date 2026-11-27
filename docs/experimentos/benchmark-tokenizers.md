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

## F10.1 Métricas y arnés

| Dimensión | Métrica | Qué mide |
| --- | --- | --- |
| Exactitud | `decode(encode(x)) == x` | Cobertura lossless |
| Compresión | tokens/carácter, CR | Eficiencia |
| Fronteras | Precision/Recall/F1 morfológica | ¿Corta tallo/sufijo donde toca? |
| Secuencia | Edit Distance, BLEU | Parecido con la referencia |
| Diversidad | NUT, LD | Uso de vocabulario |
| Modelo | BPB | Calidad del LM |
| Semántica | accuracy de tiempo verbal | ¿El modelo distingue pasado/presente/futuro? |

El arnés debe:

- construir el mismo conjunto de tokens para todos los tokenizers;
- verificar exactitud en train/test/externo;
- cachear por hash de tokenizer;
- guardar un JSON por tokenizer y un resumen.

## F10.2 Sonda de tiempo verbal

Dos niveles:

1. **Token-level**: ¿el tokenizer expone un sufijo de tiempo? Es trivial y no basta.
2. **Model-level**: entrenar el mismo LM pequeño, congelarlo y entrenar una regresión logística
   sobre el estado oculto en la posición del verbo para predecir tiempo. Comparar accuracy entre
   tokenizers.

Si el tokenizer morfológico mejora la sonda pero no BPB, sirve para interpretabilidad, no para
compresión.

## F10.3 Comparación controlada e informe

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
