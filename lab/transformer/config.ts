// Teoría: docs/teoria/secuencias/lm-transformer.md
// Experimento: contar parámetros de configuraciones candidatas para F6.

import { TransformerLM } from "../../src/models/transformer-lm.ts";

const configs = [
    { nombre: "32/4/1/64", modelSize: 32, heads: 4, blocks: 1, hiddenSize: 64 },
    { nombre: "128/8/4/512", modelSize: 128, heads: 8, blocks: 4, hiddenSize: 512 },
    { nombre: "160/8/4/640", modelSize: 160, heads: 8, blocks: 4, hiddenSize: 640 },
];

for (const config of configs) {
    const model = new TransformerLM(512, {
        modelSize: config.modelSize,
        heads: config.heads,
        blocks: config.blocks,
        hiddenSize: config.hiddenSize,
        seed: 1,
    });
    const architecture = model.architecture;
    console.log(
        `${config.nombre.padEnd(14)} ${String(model.parameterCount()).padStart(9)} parámetros  ` +
            `(d=${architecture.modelSize}, h=${architecture.heads}, ` +
            `bloques=${architecture.blocks}, ff=${architecture.hiddenSize})`,
    );
}
