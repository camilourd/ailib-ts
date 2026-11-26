// Teoría: docs/teoria/secuencias/bloque-transformer.md
// Experimento: un bloque pre-norm con dos cabezas sobre tres posiciones.

import { Tensor } from "../../src/core/tensor.ts";
import { TransformerBlock } from "../../src/nn/transformer-block.ts";

const block = new TransformerBlock(4, 2, { hiddenSize: 8, seed: 1 });
const input = Tensor.from([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [3, 4]);
const { output, weights } = block.forward(input);

const norm = (tensor: Tensor): number => {
    let total = 0;
    for (const value of tensor.data) total += value * value;
    return Math.sqrt(total);
};

console.log(`entrada ${JSON.stringify(input.shape)} · salida ${JSON.stringify(output.shape)}`);
console.log(`parámetros: ${block.parameters().length} tensores`);
console.log(`norma entrada ${norm(input).toFixed(4)} · salida ${norm(output).toFixed(4)}`);
for (let head = 0; head < weights.length; head++) {
    console.log(`cabeza ${head}:`);
    for (let row = 0; row < 3; row++) {
        const cells = [0, 1, 2].map((column) =>
            (weights[head] as Tensor).get(row, column).toFixed(2),
        );
        console.log(`  ${cells.join("  ")}`);
    }
}
