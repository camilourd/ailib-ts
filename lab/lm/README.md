# LM de caracteres

- `node lab/lm/run.ts`: entrena el LM byte-level y genera una continuación.
- `node lab/lm/vanishing.ts`: mide el gradiente por posición y genera
  `docs/teoria/secuencias/desvanecimiento.svg`.
- `node lab/lm/cells.ts`: compara RNN, GRU y LSTM y genera `gru-lstm.svg`.
- `node lab/lm/bpe.ts`: compara BPE con bytes en el mismo LM y mide BPC.
- `node lab/lm/haea.ts`: entrena RNN, GRU y LSTM con HAEA frente a SGD.

Requiere Node ≥ 23.6 (ejecuta TypeScript directamente).
