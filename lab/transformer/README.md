# Transformer

- `node lab/transformer/lm.ts`: entrena el LM transformer pequeño.
- `node lab/transformer/config.ts`: cuenta parámetros de configuraciones 1–10M.
- `node lab/transformer/pretrain.ts [semilla] [pasos]`: preentreno con schedule y checkpoints.
- `node lab/transformer/plot-pretrain.ts`: genera las curvas de preentreno.
- `node lab/transformer/plot-seeds.ts`: compara las curvas de las dos semillas.
- `node lab/transformer/benchmark.ts`: mide la inferencia baseline (tokens/s y memoria).
- `node lab/transformer/sft.ts`: SFT antes/después sobre pares del Quijote.- `node lab/transformer/ablation.ts`: quita RoPE, atención o FFN y compara BPC.

Requiere Node ≥ 23.6 (ejecuta TypeScript directamente).
