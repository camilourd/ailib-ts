// Teoría: docs/teoria/secuencias/multi-head.md
// Experimento: dos cabezas de atención sobre una secuencia de tres posiciones.

import { Tensor } from "../../src/core/tensor.ts";
import { MultiHeadAttention } from "../../src/nn/multihead.ts";

const attention = new MultiHeadAttention(4, 2, { seed: 1 });
const input = Tensor.from([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [3, 4]);
const { output, weights } = attention.forward(input);

for (let head = 0; head < weights.length; head++) {
    console.log(`cabeza ${head}:`);
    for (let row = 0; row < 3; row++) {
        const cells = [0, 1, 2].map((column) => (weights[head] as Tensor).get(row, column).toFixed(2));
        console.log(`  ${cells.join("  ")}`);
    }
}
console.log("salida:");
for (let row = 0; row < 3; row++) {
    const cells = [0, 1, 2, 3].map((column) => output.get(row, column).toFixed(3));
    console.log(`  ${cells.join("  ")}`);
}
