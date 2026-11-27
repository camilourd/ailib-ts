Modelos completos: secuencias, transformer y LLM.

`TransformerLM` acepta `normKind: "layer"|"rms"`, `ffnKind: "relu"|"swiglu"`, `parallel` y
`rope: false` para NoPE. La variante Modern es `normKind: "rms"` + `ffnKind: "swiglu"`.
