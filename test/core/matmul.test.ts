import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { matmul } from "../../src/core/tensor-ops.ts";

test("matmul coincide con el cálculo a mano", () => {
    const left = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const right = Tensor.from([7, 8, 9, 10, 11, 12], [3, 2]);
    const result = matmul(left, right);
    assert.deepEqual(result.shape, [2, 2]);
    assert.deepEqual([...result.data], [58, 64, 139, 154]);
});

test("matmul con la identidad devuelve la matriz", () => {
    const matrix = Tensor.from([1, 2, 3, 4], [2, 2]);
    const identity = Tensor.from([1, 0, 0, 1], [2, 2]);
    assert.deepEqual([...matmul(matrix, identity).data], [1, 2, 3, 4]);
    assert.deepEqual([...matmul(identity, matrix).data], [1, 2, 3, 4]);
});

test("matmul vector fila por vector columna da una matriz 1x1", () => {
    const row = Tensor.from([1, 2, 3], [1, 3]);
    const column = Tensor.from([4, 5, 6], [3, 1]);
    const result = matmul(row, column);
    assert.deepEqual(result.shape, [1, 1]);
    assert.equal(result.get(0, 0), 32);
});

test("matmul rechaza formas incompatibles", () => {
    const matrix = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const vector = Tensor.from([1, 2, 3], [3]);
    assert.throws(() => matmul(matrix, vector), Error);
    assert.throws(() => matmul(vector, vector), Error);
    const wrong = Tensor.from([1, 2, 3, 4], [2, 2]);
    assert.throws(() => matmul(matrix, wrong), Error);
});
