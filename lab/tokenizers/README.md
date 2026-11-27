# Tokenizers

- `node lab/tokenizers/baselines.ts`: byte/carácter/palabra/BPE 512/1024/2048 sobre Quijote y Lazarillo.
- `node lab/tokenizers/program-check.ts`: genoma mínimo lossless (palabra/carácter) sobre todo el corpus.
- `node lab/tokenizers/pilot.ts`: F9.2, 200 aleatorios + SGE 20×10 con fitness barato.
- `node lab/tokenizers/sweep.ts`: F9.3, 100×100 SGE, 5 genes y barrido por vocabulario.
- `node lab/tokenizers/context.ts`: F9.3a, DP contextual con pasado/futuro.
- `node lab/tokenizers/semantic.ts`: F9.3b, terminaciones verbales y etiquetas con BPB proxy.
- `node lab/tokenizers/sweep-v2.ts`: F9.3c, 100×100 con SGE/GE/πGE/HAEA sobre 480 combinaciones.
- `node lab/tokenizers/lm-quick.ts`: F9.4 proxy, BPE vs substring/palabra/carácter en BPB.
- `node lab/tokenizers/export-gpu.ts`: F9.4b exporta BPE y BPE+sufijos para GPU.
- `node lab/tokenizers/lm-hybrid-quick.ts`: F9.4b proxy BPE vs BPE+sufijos.
- `python lab/tokenizers/train_gpu.py`: F9.4b, LM 1.4M en GPU (requiere CUDA).

Requiere Node ≥ 23.6 (ejecuta TypeScript directamente).
