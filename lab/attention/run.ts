// Teoría: docs/teoria/secuencias/atencion.md
// Experimento: atención causal sobre una secuencia corta y sus pesos.

import { Tensor } from "../../src/core/tensor.ts";
import { scaledDotProductAttention } from "../../src/nn/attention.ts";

const queries = Tensor.from([1, 0, 0, 1, 1, 1, 0, 1], [4, 2]);
const keys = Tensor.from([1, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
const values = Tensor.from([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
const { output, weights } = scaledDotProductAttention(queries, keys, values);

console.log("pesos de atención (filas = consulta, columnas = clave):");
for (let row = 0; row < 4; row++) {
    const cells = [0, 1, 2, 3].map((column) => weights.get(row, column).toFixed(2));
    console.log(`  ${cells.join("  ")}`);
}
console.log("salida:");
for (let row = 0; row < 4; row++) {
    console.log(`  ${output.get(row, 0).toFixed(3)}  ${output.get(row, 1).toFixed(3)}`);
}
