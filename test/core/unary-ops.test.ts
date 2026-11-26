import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { exp, log, relu, sum, tanh } from "../../src/core/tensor-ops.ts";

const tracked = (values: number[]): Tensor =>
    Tensor.from(values, [values.length], { requiresGrad: true });

test("exp y log son inversas", () => {
    const input = Tensor.from([0, 1, 2], [3]);
    const output = log(exp(input));
    [0, 1, 2].forEach((expected, index) => {
        assert.ok(Math.abs((output.data[index] as number) - expected) < 1e-5);
    });
});

test("exp: el gradiente es e^x", () => {
    const input = tracked([0, 1]);
    sum(exp(input)).backward();
    assert.ok(Math.abs((input.grad?.data[0] ?? 0) - 1) < 1e-6);
    assert.ok(Math.abs((input.grad?.data[1] ?? 0) - Math.E) < 1e-5);
});

test("log: el gradiente es 1/x", () => {
    const input = tracked([1, 2, 4]);
    sum(log(input)).backward();
    [1, 0.5, 0.25].forEach((expected, index) => {
        assert.ok(Math.abs((input.grad?.data[index] ?? 0) - expected) < 1e-6);
    });
});

test("relu recorta los negativos y su gradiente es 0 o 1", () => {
    const input = tracked([-2, 0, 3]);
    const output = relu(input);
    assert.deepEqual([...output.data], [0, 0, 3]);
    sum(output).backward();
    assert.deepEqual([...(input.grad?.data ?? [])], [0, 0, 1]);
});

test("tanh: valores y derivada", () => {
    const input = tracked([-1, 0, 1]);
    const output = tanh(input);
    [-1, 0, 1].forEach((value, index) => {
        assert.ok(Math.abs((output.data[index] as number) - Math.tanh(value)) < 1e-6);
    });
    sum(output).backward();
    [-1, 0, 1].forEach((value, index) => {
        const expected = 1 - Math.tanh(value) ** 2;
        assert.ok(Math.abs((input.grad?.data[index] ?? 0) - expected) < 1e-6);
    });
});
