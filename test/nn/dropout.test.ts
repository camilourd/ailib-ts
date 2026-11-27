import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { Dropout } from "../../src/nn/dropout.ts";

test("Dropout no toca en evaluación y apaga en entrenamiento", () => {
    const layer = new Dropout({ rate: 0.5, seed: 1 });
    const input = Tensor.ones([100], { requiresGrad: true });
    const evaluation = layer.forward(input);
    assert.equal(evaluation, input);
    const training = layer.forward(input, { training: true });
    assert.equal(training.size, 100);
    for (const value of training.data) assert.ok(value === 0 || value === 2);
    assert.equal(layer.parameters().length, 0);
});

test("Dropout propaga gradiente con la misma máscara", () => {
    const layer = new Dropout({ rate: 0.25, seed: 2 });
    const input = Tensor.ones([20], { requiresGrad: true });
    const output = layer.forward(input, { training: true });
    sum(output).backward();
    for (let index = 0; index < input.size; index++) {
        const expected = output.data[index] === 0 ? 0 : 1 / 0.75;
        assert.ok(Math.abs((input.grad?.data[index] as number) - expected) < 1e-6);
    }
    assert.throws(() => new Dropout({ rate: 1 }), RangeError);
});
