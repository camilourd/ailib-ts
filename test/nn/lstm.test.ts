import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { LSTM } from "../../src/nn/lstm.ts";

test("LSTM: estado inicial, paso y avance", () => {
    const lstm = new LSTM(3, 4, { seed: 1 });
    const state = lstm.initial();
    assert.deepEqual(state.hidden.shape, [1, 4]);
    assert.deepEqual(state.cell?.shape, [1, 4]);
    const input = Tensor.from([1, 2, 3], [1, 3], { requiresGrad: true });
    const next = lstm.step(input, state.hidden, state.cell as Tensor);
    assert.deepEqual(next.hidden.shape, [1, 4]);
    assert.deepEqual(next.cell?.shape, [1, 4]);
    const advanced = lstm.advance(input, state);
    assert.deepEqual(advanced.hidden.shape, [1, 4]);
    assert.equal(lstm.parameters().length, 12);
});

test("LSTM: el gradiente llega a parámetros y entrada", () => {
    const lstm = new LSTM(2, 3, { seed: 1 });
    const input = Tensor.from([1, 2], [1, 2], { requiresGrad: true });
    const state = lstm.initial();
    sum(lstm.step(input, state.hidden, state.cell as Tensor).hidden).backward();
    for (const parameter of lstm.parameters()) assert.notEqual(parameter.grad, null);
    assert.notEqual(input.grad, null);
});

test("LSTM: valida formas", () => {
    const lstm = new LSTM(2, 3);
    const state = lstm.initial();
    assert.throws(() => lstm.step(Tensor.from([1], [1]), state.hidden, state.cell as Tensor), Error);
    assert.throws(() => lstm.step(Tensor.from([1, 2], [1, 2]), state.hidden, Tensor.zeros([1, 2])), Error);
});
