import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { GRU } from "../../src/nn/gru.ts";

test("GRU: estado inicial, paso y avance", () => {
    const gru = new GRU(3, 4, { seed: 1 });
    const state = gru.initial();
    assert.deepEqual(state.hidden.shape, [1, 4]);
    assert.deepEqual([...state.hidden.data], [0, 0, 0, 0]);
    const input = Tensor.from([1, 2, 3], [1, 3], { requiresGrad: true });
    const next = gru.step(input, state.hidden);
    assert.deepEqual(next.shape, [1, 4]);
    const advanced = gru.advance(input, state);
    assert.deepEqual(advanced.hidden.shape, [1, 4]);
    assert.equal(gru.parameters().length, 9);
});

test("GRU: el gradiente llega a parámetros y entrada", () => {
    const gru = new GRU(2, 3, { seed: 1 });
    const input = Tensor.from([1, 2], [1, 2], { requiresGrad: true });
    sum(gru.step(input, gru.initial().hidden)).backward();
    for (const parameter of gru.parameters()) assert.notEqual(parameter.grad, null);
    assert.notEqual(input.grad, null);
});

test("GRU: valida formas", () => {
    const gru = new GRU(2, 3);
    assert.throws(() => gru.step(Tensor.from([1], [1]), gru.initial().hidden), Error);
    assert.throws(() => gru.step(Tensor.from([1, 2], [1, 2]), Tensor.zeros([1, 2])), Error);
});
