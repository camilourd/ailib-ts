import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { MLP } from "../../src/nn/mlp.ts";

test("MLP: forma de salida y número de parámetros", () => {
    const mlp = new MLP([3, 4, 1], { seed: 1 });
    const input = Tensor.from(
        Array.from({ length: 12 }, (_, index) => index / 12),
        [4, 3],
    );
    const output = mlp.forward(input);
    assert.deepEqual(output.shape, [4, 1]);
    assert.equal(mlp.parameters().length, 4);
});

test("MLP: el gradiente llega a todos los parámetros", () => {
    const mlp = new MLP([2, 3, 1], { seed: 1 });
    const input = Tensor.from([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    sum(mlp.forward(input)).backward();
    for (const parameter of mlp.parameters()) {
        assert.notEqual(parameter.grad, null);
    }
    assert.notEqual(input.grad, null);
});

test("MLP: tamaños inválidos fallan", () => {
    assert.throws(() => new MLP([2]), Error);
    assert.throws(() => new MLP([2, 0, 1]), Error);
    assert.throws(() => new MLP([2, 1.5]), Error);
});
