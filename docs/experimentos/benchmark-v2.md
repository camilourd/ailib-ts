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

Para cada dominio:

- BPE, SentencePiece-BPE, WordPiece y Unigram.
- Vocabulario objetivo 2k/8k/16k.
- Métricas: exactitud, vocabulario real, tokens/carácter y, en narrativo, frontera/ED.
- Sin LM: primero decidir qué tokenizers merecen GPU.

Comando: `node lab/benchmark/tokenizer_zoo.ts`.

Resultado parcial (narrativo, BPE 2k): real 2.048, 0,284 tokens/carácter, exacto.

## Límites

- Técnico y código son del propio repo; no son un corpus externo independiente.
- WordPiece/Unigram pueden no alcanzar el vocabulario objetivo.
- La comparación de dominios no es de tamaño igual: el código tiene más caracteres.
- El zoo completo es caro; 16k puede tardar.

## Siguiente

- Terminar el zoo y elegir finalistas por dominio.
- Pasar solo esos a F12.2 (LM 5,8M, mismas semillas/presupuesto).
- Añadir una tarea downstream por dominio.
