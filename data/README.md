# Corpus

`quijote.txt` es un extracto de **Don Quijote de la Mancha**, de Miguel de Cervantes, a partir del
texto del Proyecto Gutenberg (eBook #2000), de dominio público.

- Fuente: `https://www.gutenberg.org/ebooks/2000`
- Copia de trabajo: `https://raw.githubusercontent.com/GITenberg/Don-Quijote_2000/master/2000-8.txt`
- El original está en ISO-8859-1; `lab/lm/prepare-quijote.mjs` lo convierte a UTF-8 y recorta los
  primeros 200.000 caracteres de la novela.
- El archivo completo descargado (`quijote-full.txt`) no se versiona.
- `quijote-bpe.json`: merges del BPE entrenado con el 90 % inicial y vocabulario 512 (256 merges);
  0,434 tokens/carácter. Los scripts están en `lab/lm/prepare-quijote.mjs` y `lab/lm/train-bpe.ts`.
- `pretrain-seed-1.json`: curvas del preentreno de la semilla 1; los checkpoints van a `checkpoints/`
  y no se versionan.
- `pretrain-seed-2.json`: curvas de la semilla 2 para comparar.
- `sft-quijote.json`: 16 pares de train y 6 de validación para el SFT.
- `sft-report.json`: pérdidas, perplejidades y muestras antes/después del SFT.
- `inference-baseline.json`: tokens/s, latencia y memoria de la generación sin caché.
- `inference-kv-cache.json`: comparación sin caché vs KV cache, con salida idéntica.
- `sampling-report.json`: muestras con temperatura, top-k y top-p.
- `quantization-report.json`: tamaño y perplejidad de float32, INT8 por canal y por tensor.
- `selective-quantization.json`: variantes por módulo y su perplejidad.
- `tokenizers-report.json`: comparación byte, carácter, BPE y palabra en BPB.
- `lazarillo.txt`: texto externo de *Lazarillo de Tormes* (Gutenberg #320) para el holdout.
- `tokenizer-baselines.json`: F9.0, baselines exactos sobre Quijote y Lazarillo.
- `tokenizer-program.json`: F9.1, genoma mínimo lossless (palabra/carácter).
- `tokenizer-pilot.json`: F9.2, 200 aleatorios + SGE 20×10 sobre 20k/8k/10k.
- `tokenizer-sweep.json`: F9.3, 100×100 SGE, 358 únicos y mejor por vocabulario.
- `tokenizer-context.json`: F9.3a, DP contextual (resultado negativo: mismo tokens que sin contexto).
- `tokenizer-semantic.json`: F9.3b, base vs terminaciones vs etiquetas con BPB proxy.
- `tokenizer-sweep-v2.json`: F9.3c, 480 combinaciones y cuatro operadores; todos empatan en 0,429.
- `tokenizer-lm-quick.json`: F9.4 proxy, BPB de BPE/substring/palabra/carácter.
- `tokenizer-hybrid-quick.json`: F9.4b proxy, BPE vs BPE+sufijos.
- `tokenizer-gpu.json`: F9.4b GPU, BPE vs prefijos/sufijos/números en BPB.
- `tokenizer-gpu-8m.json`: F11.4 comparación GPU ~5,8M de familias a 2k.
- `tokenizer-gpu-hybrids.json`: F11.5 híbridos BPE+prefijo/sufijo/números.
- `tokenizer-gpu-fair-complete.json`: F11.5 tabla completa familias + híbridos.
- `tokenizer-vocab-sweep.json`: F11.6a barrido 2k/8k/16k con vocabulario real.
- `tokenizer-gpu-var-small.json`: F11.6b variación 1,9M/100 pasos.
- `tokenizer-gpu-var-context64.json`: F11.6b variación 5,8M/100/contexto64.
- `tokenizer-gpu-variations.json`: F11.6b tabla completa de variaciones 1,9M/contexto.
- `tokenizer-gpu-arch.json`: F11.6b pre/post-norm, GRU y LSTM.
- `tokenizer-gpu-arch-variants.json`: F11.6b modern, NoPE y paralelo.
- `tokenizer-gpu-modern-bpe-400.json`: F11.6b Modern 400 con las 4 variantes BPE.
- `morphology-reference.json`: F10.0, 800 formas verbales finitas con tallo/sufijo/tiempo.
- `morphology-review.csv`: F10.0, 100 formas para revisión humana.
- `morphology-reference-reviewed.json`: F10.0, 795 formas tras revisar 100 decisiones.
- `morphology-review-fixed.csv`: F10.0, CSV corregido y auditado.
- `morphology-review-audit.json`, `morphology-review-decisions.json`: F10.0, auditoría y decisiones.
- `tokenizer-benchmark.json`: F10.1, métricas multidimensionales de los tokenizers.
- `tense-probe.json`: F10.2, accuracy de tiempo verbal con LM congelado.
- `tokenizer-evolution-f10.json`: F10.3, evolución con fitness frontera/ED.

El extracto se usa solo para entrenar el LM pequeño de F6.
