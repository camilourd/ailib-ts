// Teoría: docs/teoria/nucleo/tensor.md
// Experimento: recorre un tensor 2D, lo transpone y lo reordena, mostrando forma y strides.

import { Tensor } from "../../src/core/tensor.ts";

function printMatrix(label: string, tensor: Tensor): void {
    console.log(`${label}: forma ${JSON.stringify(tensor.shape)} · strides ${JSON.stringify(tensor.strides)}`);
    const rows = tensor.shape[0] as number;
    const columns = tensor.shape[1] as number;
    for (let row = 0; row < rows; row++) {
        const values = Array.from({ length: columns }, (_, column) => tensor.get(row, column));
        console.log(values.map((value) => String(value).padStart(4)).join(" "));
    }
}

const matrix = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
printMatrix("original", matrix);
printMatrix("transpuesta", matrix.transpose());
printMatrix("reshape 3x2", matrix.reshape([3, 2]));
