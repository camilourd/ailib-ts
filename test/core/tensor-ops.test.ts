import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { add, broadcastShapes, broadcastStrides, mul } from "../../src/core/tensor-ops.ts";

test("broadcastShapes aplica las reglas", () => {
    assert.deepEqual(broadcastShapes([2, 3], [3]), [2, 3]);
    assert.deepEqual(broadcastShapes([2, 1], [1, 3]), [2, 3]);
    assert.deepEqual(broadcastShapes([], [2, 3]), [2, 3]);
    assert.deepEqual(broadcastShapes([1], [2, 3]), [2, 3]);
    assert.throws(() => broadcastShapes([2, 3], [2]), Error);
    assert.throws(() => broadcastShapes([2, 3], [4, 3]), Error);
});

test("broadcastStrides pone 0 en las dimensiones expandidas", () => {
    assert.deepEqual(broadcastStrides([3], [1], [2, 3]), [0, 1]);
    assert.deepEqual(broadcastStrides([2, 1], [1, 1], [2, 3]), [1, 0]);
    assert.deepEqual(broadcastStrides([], [], [2, 3]), [0, 0]);
    assert.deepEqual(broadcastStrides([2, 3], [3, 1], [2, 3]), [3, 1]);
});

test("add suma con broadcasting", () => {
    const batch = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const bias = Tensor.from([10, 20, 30], [3]);
    const result = add(batch, bias);
    assert.deepEqual(result.shape, [2, 3]);
    assert.deepEqual([...result.data], [11, 22, 33, 14, 25, 36]);
});

test("mul multiplica con broadcasting", () => {
    const column = Tensor.from([1, 2], [2, 1]);
    const row = Tensor.from([10, 20, 30], [1, 3]);
    const result = mul(column, row);
    assert.deepEqual(result.shape, [2, 3]);
    assert.deepEqual([...result.data], [10, 20, 30, 20, 40, 60]);
});

test("un escalar se combina con cualquier forma", () => {
    const scalar = Tensor.from([2], []);
    const vector = Tensor.from([1, 2, 3], [3]);
    assert.deepEqual([...mul(scalar, vector).data], [2, 4, 6]);
    assert.deepEqual([...add(scalar, vector).data], [3, 4, 5]);
});

test("formas incompatibles fallan", () => {
    const matrix = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const bad = Tensor.from([1, 2], [2]);
    assert.throws(() => add(matrix, bad), Error);
    assert.throws(() => mul(bad, matrix), Error);
});
