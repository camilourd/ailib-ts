import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { Linear } from "../../src/nn/linear.ts";

test("Linear calcula y = xW + b", () => {
    const layer = new Linear(2, 2, { seed: 1 });
    layer.weights.data.set([1, 2, 3, 4]);
    layer.bias.data.set([10, 20]);
    const input = Tensor.from([1, 2], [1, 2]);
    const output = layer.forward(input);
    assert.deepEqual(output.shape, [1, 2]);
    assert.deepEqual([...output.data], [17, 30]);
});

test("la inicialización es reproducible por semilla y está acotada", () => {
    const first = new Linear(3, 4, { seed: 7 });
    const second = new Linear(3, 4, { seed: 7 });
    assert.deepEqual([...first.weights.data], [...second.weights.data]);
    const limit = Math.sqrt(6 / (3 + 4));
    for (const value of first.weights.data) {
        assert.ok(Math.abs(value) <= limit + 1e-6, `${value} fuera de ±${limit}`);
    }
    assert.deepEqual([...first.bias.data], [0, 0, 0, 0]);
});

test("los parámetros piden gradiente y el gradiente llega a todos", () => {
    const layer = new Linear(2, 3, { seed: 1 });
    assert.deepEqual(layer.parameters().map((parameter) => parameter.shape), [[2, 3], [3]]);
    const input = Tensor.from([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const output = layer.forward(input);
    sum(output).backward();
    assert.deepEqual(layer.weights.grad?.shape, [2, 3]);
    assert.deepEqual(layer.bias.grad?.shape, [3]);
    assert.deepEqual(input.grad?.shape, [2, 2]);
});

test("una entrada con forma inválida falla", () => {
    const layer = new Linear(2, 3);
    assert.throws(() => layer.forward(Tensor.from([1, 2, 3], [3])), Error);
    assert.throws(() => layer.forward(Tensor.from([1, 2, 3, 4], [1, 4])), Error);
});

test("tamaños de capa inválidos fallan", () => {
    assert.throws(() => new Linear(0, 2), Error);
    assert.throws(() => new Linear(2, -1), Error);
    assert.throws(() => new Linear(2.5, 2), Error);
});
