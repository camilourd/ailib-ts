import assert from "node:assert/strict";
import { test } from "node:test";

import { sum } from "../../src/core/tensor-ops.ts";
import { Embedding } from "../../src/nn/embedding.ts";

test("forward busca las filas de la tabla", () => {
    const embedding = new Embedding(3, 2, { seed: 1 });
    embedding.weights.data.set([1, 2, 3, 4, 5, 6]);
    const output = embedding.forward([0, 2]);
    assert.deepEqual(output.shape, [2, 2]);
    assert.deepEqual([...output.data], [1, 2, 5, 6]);
    assert.deepEqual([...embedding.row(1).data], [3, 4]);
});

test("el backward reparte el gradiente en las filas usadas", () => {
    const embedding = new Embedding(3, 2, { seed: 1 });
    embedding.weights.data.set([1, 2, 3, 4, 5, 6]);
    sum(embedding.forward([0, 2, 0])).backward();
    assert.deepEqual([...(embedding.weights.grad?.data ?? [])], [2, 2, 0, 0, 1, 1]);
});

test("la misma semilla da la misma tabla", () => {
    const first = new Embedding(4, 3, { seed: 7 });
    const second = new Embedding(4, 3, { seed: 7 });
    assert.deepEqual([...first.weights.data], [...second.weights.data]);
});

test("valida vocabulario, dimensión e índices", () => {
    assert.throws(() => new Embedding(0, 2), Error);
    assert.throws(() => new Embedding(2, 0), Error);
    const embedding = new Embedding(2, 2);
    assert.throws(() => embedding.forward([2]), RangeError);
    assert.throws(() => embedding.forward([-1]), RangeError);
    assert.throws(() => embedding.forward([0.5]), RangeError);
});
