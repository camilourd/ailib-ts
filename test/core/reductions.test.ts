import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { mean, sum } from "../../src/core/tensor-ops.ts";

const matrix = (): Tensor => Tensor.from([1, 2, 3, 4], [2, 2]);

test("sum sin eje suma todo y devuelve un escalar", () => {
    const result = sum(matrix());
    assert.deepEqual(result.shape, []);
    assert.equal(result.get(), 10);
});

test("sum por eje elimina ese eje", () => {
    const columns = sum(matrix(), 0);
    assert.deepEqual(columns.shape, [2]);
    assert.deepEqual([...columns.data], [4, 6]);
    const rows = sum(matrix(), 1);
    assert.deepEqual(rows.shape, [2]);
    assert.deepEqual([...rows.data], [3, 7]);
});

test("mean sin eje y por eje", () => {
    assert.equal(mean(matrix()).get(), 2.5);
    assert.deepEqual([...mean(matrix(), 0).data], [2, 3]);
    assert.deepEqual([...mean(matrix(), 1).data], [1.5, 3.5]);
});

test("sum y mean funcionan con un escalar", () => {
    const scalar = Tensor.from([5], []);
    assert.equal(sum(scalar).get(), 5);
    assert.equal(mean(scalar).get(), 5);
});

test("un eje inválido falla", () => {
    assert.throws(() => sum(matrix(), 2), RangeError);
    assert.throws(() => sum(matrix(), -1), RangeError);
    assert.throws(() => mean(matrix(), 0.5), RangeError);
});

test("un eje de tamaño 0 da un resultado vacío o de ceros", () => {
    const empty = Tensor.zeros([0, 3]);
    const columns = sum(empty, 0);
    assert.deepEqual(columns.shape, [3]);
    assert.deepEqual([...columns.data], [0, 0, 0]);
    const rows = sum(empty, 1);
    assert.deepEqual(rows.shape, [0]);
    assert.equal(rows.size, 0);
});
