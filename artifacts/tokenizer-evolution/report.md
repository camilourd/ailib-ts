# Tokenizers sin pérdida: informe F9

**Estado:** cerrado · **Repositorio:** ailib-ts · **Fecha:** 2026-11-27

## Pregunta

¿Se puede evolucionar un tokenizer lossless que comprima mejor que los baselines y que además ayude
al LM?

## Qué se construyó

- `LosslessByteTokenizer`: bytes + tokens compuestos, decode exacto.
- `ContextTokenizer`: segmentación por DP y contexto izquierdo/derecho.
- Genoma de tokenizers: fuente, vocabulario, `maxLen`, frecuencia, política, contexto y semántica.
- Baselines: byte, carácter, palabra, BPE 2048.
- Evaluador GPU: `lab/tokenizers/train_gpu.py` (PyTorch/CUDA, GTX 1060 3GB).
- Exportadores: `export-gpu.ts`, `export-gpu-shape.ts`.

## Compresión (tokens/carácter)

| Tokenizer | Validación | Externo |
| --- | ---: | ---: |
| Mejor F9.3 (`substring/16384/32/1`) | **0,429** | **0,471** |
| BPE 2048 (Quijote completo, referencia F9.0) | 0,285 | 0,313 |
| Word | 0,569 | — |
| Char | 1,000 | — |

El mejor evolucionado **no supera a BPE** en compresión cuando se compara sobre el mismo corpus.

## Contexto y semántica

| Idea | Tokens | BPB proxy | Conclusión |
| --- | --- | --- | --- |
| Contexto 2/2, lookahead | 0,276 | — | Igual que sin contexto; el DP ya es óptimo |
| Sufijos verbales | 0,484 | 4,179 | Peor |
| Etiquetas de tipo | 0,558 | 4,807 | Peor |
| Sufijos (post-BPE, GPU) | 2,3999 test / 2,6932 ext | — | Ruido, no mejora |
| Prefijos + sufijos (GPU) | 2,4027 / 2,6938 | — | No mejora |
| Mayúsculas (GPU) | **2,3840** / 2,6833 | — | Gana test, pierde externo |
| Forma de palabra (GPU) | 2,4060 / 2,6783 | — | No mejora |
| BPE dropout 10% (GPU) | 2,5044 / 2,7723 | — | Peor |

## Ganador

**BPE 2048** sigue siendo el mejor en el externo: **2,6752 BPB**. Ninguna de las variantes
morfológicas o de forma lo supera de manera consistente. La variante de mayúsculas mejora el test
(2,3840) pero empeora Lazarillo (2,6833), señal de sobreajuste al Quijote.

## Lo aprendido

1. **Menos tokens no es mejor modelo.** El tokenizer evolucionado comprime más que BPE en algunos
   cortes, pero su BPB es peor.
2. **El contexto no cambia el óptimo de tokens** con un vocabulario fijo: el DP ya elige el mínimo.
3. **La semántica añade tokens**; solo paga si el LM es más grande o si hay una tarea que la use.
4. **Los métodos evolutivos empatan** porque el espacio de 480 combinaciones se agota; faltan genes
   de composición (split, híbridos, BPE) para que la búsqueda tenga algo que explorar.
5. **El baseline es fuerte.** BPE 2048 es difícil de batir con este presupuesto.

## Reproducir

```bash
node lab/tokenizers/context.ts
node lab/tokenizers/semantic.ts
node lab/tokenizers/sweep-v2.ts
node lab/tokenizers/export-gpu.ts
node lab/tokenizers/export-gpu-shape.ts
python lab/tokenizers/train_gpu.py
```

Datos: `data/tokenizer-context.json`, `tokenizer-semantic.json`, `tokenizer-sweep-v2.json`,
`tokenizer-lm-quick.json`, `tokenizer-hybrid-quick.json`, `tokenizer-gpu.json`.

## Próximo

- LM más grande (≥10M) y más datos.
- Genes de composición: split + híbrido + BPE.
- Tokens de código/URL si el corpus lo pide.
- Evaluar morfología con una tarea explícita de tiempo verbal, no solo BPB.
