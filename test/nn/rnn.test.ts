import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { RNN } from "../../src/nn/rnn.ts";

test("RNN: estado inicial, paso y secuencia", () => {
    const rnn = new RNN(3, 4, { seed: 1 });
    assert.equal(rnn.inputSize, 3);
    assert.equal(rnn.hiddenSize, 4);

    const zero = rnn.initialHidden();
    assert.deepEqual(zero.shape, [1, 4]);
    assert.deepEqual([...zero.data], [0, 0, 0, 0]);
    assert.equal(zero.requiresGrad, false);

    const input = Tensor.from([1, 2, 3], [1, 3]);
    const next = rnn.step(input, zero);
    assert.deepEqual(next.shape, [1, 4]);

    const states = rnn.forward([input, input, input]);
    assert.equal(states.length, 3);
    for (const state of states) assert.deepEqual(state.shape, [1, 4]);
    assert.equal(rnn.parameters().length, 3);
});

test("RNN: el gradiente llega a parámetros y a la entrada", () => {
    const rnn = new RNN(2, 3, { seed: 1 });
    const inputs = [
        Tensor.from([1, 2], [1, 2], { requiresGrad: true }),
        Tensor.from([3, 4], [1, 2], { requiresGrad: true }),
    ];
    const states = rnn.forward(inputs);
    sum(states[1] as Tensor).backward();
    for (const parameter of rnn.parameters()) {
        assert.notEqual(parameter.grad, null);
    }
    assert.notEqual(inputs[0]?.grad, null);
});

test("RNN: la misma semilla da los mismos pesos", () => {
    const first = new RNN(2, 3, { seed: 5 });
    const second = new RNN(2, 3, { seed: 5 });
    for (let index = 0; index < first.parameters().length; index++) {
        assert.deepEqual(
            [...(first.parameters()[index]?.data ?? [])],
            [...(second.parameters()[index]?.data ?? [])],
        );
    }
});

test("RNN: valida entrada, estado y tamaños", () => {
    const rnn = new RNN(2, 3);
    const input = Tensor.from([1, 2], [1, 2]);
    const hidden = Tensor.zeros([1, 3]);
    assert.throws(() => rnn.step(Tensor.from([1], [1]), hidden), Error);
    assert.throws(() => rnn.step(input, Tensor.zeros([1, 2])), Error);
    assert.throws(() => new RNN(0, 3), Error);
    assert.throws(() => new RNN(2, 0), Error);
});
