// Teoría: docs/teoria/redes/capa-lineal.md
// Experimento: una capa densa 2 → 3 sobre un lote de dos vectores.

import { Tensor } from "../../src/core/tensor.ts";
import { Linear } from "../../src/nn/linear.ts";

const layer = new Linear(2, 3, { seed: 1 });
const input = Tensor.from([1, 2, 3, 4], [2, 2]);
const output = layer.forward(input);

console.log(`entrada [2, 2] · pesos ${JSON.stringify(layer.weights.shape)} · salida ${JSON.stringify(output.shape)}`);
for (let row = 0; row < 2; row++) {
    const values = [0, 1, 2].map((column) => output.get(row, column).toFixed(4));
    console.log(`  fila ${row}: ${values.join("  ")}`);
}
