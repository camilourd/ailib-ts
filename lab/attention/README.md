# Atención

- `node lab/attention/run.ts`: atención causal sobre cuatro posiciones.
- `node lab/attention/multihead.ts`: dos cabezas sobre tres posiciones.
- `node lab/attention/block.ts`: un bloque pre-norm completo.
- `node lab/attention/rope.ts`: rotación por posición y producto relativo.
- `node lab/attention/plot.ts`: genera `docs/teoria/secuencias/multi-head.svg`.
- `node lab/attention/plot-block.ts`: genera `docs/teoria/secuencias/bloque-transformer.svg`.
- `node lab/attention/plot-rope.ts`: genera `docs/teoria/secuencias/rope.svg`.

El LM transformer está en `lab/transformer/`.

Requiere Node ≥ 23.6 (ejecuta TypeScript directamente).
