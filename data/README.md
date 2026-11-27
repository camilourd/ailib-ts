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

El extracto se usa solo para entrenar el LM pequeño de F6.
