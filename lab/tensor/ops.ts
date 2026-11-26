// Teoría: docs/teoria/nucleo/broadcasting.md
// Experimento: suma un sesgo a un lote y multiplica una columna por una fila, con broadcasting.

import { Tensor } from "../../src/core/tensor.ts";
import { add, matmul, mean, mul, sum } from "../../src/core/tensor-ops.ts";

function printTensor(label: string, tensor: Tensor): void {
    console.log(`${label}: forma ${JSON.stringify(tensor.shape)} · strides ${JSON.stringify(tensor.strides)}`);
    if (tensor.rank === 2) {
        const rows = tensor.shape[0] as number;
        const columns = tensor.shape[1] as number;
        for (let row = 0; row < rows; row++) {
            const values = Array.from({ length: columns }, (_, column) => tensor.get(row, column));
            console.log(values.map((value) => String(value).padStart(4)).join(" "));
        }
        return;
    }
    console.log([...tensor.data].map((value) => String(value).padStart(4)).join(" "));
}

const batch = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
const bias = Tensor.from([10, 20, 30], [3]);
printTensor("lote", batch);
printTensor("sesgo", bias);
printTensor("lote + sesgo", add(batch, bias));

const column = Tensor.from([1, 2], [2, 1]);
const row = Tensor.from([10, 20, 30], [1, 3]);
printTensor("columna * fila", mul(column, row));

const left = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
const right = Tensor.from([7, 8, 9, 10, 11, 12], [3, 2]);
printTensor("producto matricial", matmul(left, right));

const values = Tensor.from([1, 2, 3, 4], [2, 2]);
printTensor("suma total", sum(values));
printTensor("suma por columna", sum(values, 0));
printTensor("media por fila", mean(values, 1));
