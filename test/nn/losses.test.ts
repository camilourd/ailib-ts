import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { crossEntropy, mse } from "../../src/nn/losses.ts";

function numericGradient(f: (tensor: Tensor) => Tensor, tensor: Tensor, epsilon = 1e-2): Float32Array {
    const original = tensor.data.slice();
    const result = new Float32Array(tensor.size);
    for (let index = 0; index < tensor.size; index++) {
        tensor.data[index] = (original[index] as number) + epsilon;
        const plus = f(tensor).get();
        tensor.data[index] = (original[index] as number) - epsilon;
        const minus = f(tensor).get();
        tensor.data[index] = original[index] as number;
        result[index] = (plus - minus) / (2 * epsilon);
    }
    return result;
}

test("mse: valor y gradiente", () => {
    const prediction = Tensor.from([1, 2], [2], { requiresGrad: true });
    const target = Tensor.from([0, 0], [2], { requiresGrad: true });
    const loss = mse(prediction, target);
    assert.equal(loss.get(), 2.5);
    loss.backward();
    assert.deepEqual([...(prediction.grad?.data ?? [])], [1, 2]);
    assert.deepEqual([...(target.grad?.data ?? [])], [-1, -2]);
});

test("mse coincide con la derivada numérica", () => {
    const prediction = Tensor.from([0.5, -1, 2], [3], { requiresGrad: true });
    const target = Tensor.from([0, 0, 0], [3]);
    mse(prediction, target).backward();
    const numeric = numericGradient((tensor) => mse(tensor, target), prediction);
    for (let index = 0; index < prediction.size; index++) {
        const analytic = (prediction.grad?.data[index] ?? 0) as number;
        const expected = numeric[index] as number;
        assert.ok(Math.abs(analytic - expected) < 1e-2, `en ${index}: ${analytic} vs ${expected}`);
    }
});

test("mse exige formas iguales", () => {
    assert.throws(() => mse(Tensor.from([1], [1]), Tensor.from([1, 2], [2])), Error);
});

test("crossEntropy: ejemplo a mano", () => {
    const logits = Tensor.from([1, 2, 3], [1, 3], { requiresGrad: true });
    const loss = crossEntropy(logits, [2]);
    assert.ok(Math.abs(loss.get() - 0.40760596) < 1e-5);
    loss.backward();
    const expected = [0.09003057, 0.24472847, -0.33475904];
    expected.forEach((value, index) => {
        assert.ok(Math.abs((logits.grad?.data[index] ?? 0) - value) < 1e-5, `en ${index}`);
    });
});

test("crossEntropy: media del lote", () => {
    const logits = Tensor.from([0, 0, 10, 0], [2, 2], { requiresGrad: true });
    const loss = crossEntropy(logits, [0, 1]);
    assert.ok(Math.abs(loss.get() - (Math.log(2) + 10.0000454) / 2) < 1e-4);
    loss.backward();
    const expected = [-0.25, 0.25, 0.4999773, -0.4999773];
    expected.forEach((value, index) => {
        assert.ok(Math.abs((logits.grad?.data[index] ?? 0) - value) < 1e-4, `en ${index}`);
    });
});

test("crossEntropy es estable con logits grandes", () => {
    const logits = Tensor.from([1000, 1001], [1, 2], { requiresGrad: true });
    const loss = crossEntropy(logits, [1]);
    assert.ok(Number.isFinite(loss.get()));
    assert.ok(Math.abs(loss.get() - Math.log(1 + Math.exp(-1))) < 1e-6);
    loss.backward();
    assert.ok((logits.grad?.data ?? []).every((value) => Number.isFinite(value)));
});

test("crossEntropy coincide con la derivada numérica", () => {
    const logits = Tensor.from([0.5, -0.2, 1.1, 0.3, 0.7, -0.4], [2, 3], { requiresGrad: true });
    const targets = [2, 0];
    crossEntropy(logits, targets).backward();
    const numeric = numericGradient((tensor) => crossEntropy(tensor, targets), logits);
    for (let index = 0; index < logits.size; index++) {
        const analytic = (logits.grad?.data[index] ?? 0) as number;
        const expected = numeric[index] as number;
        assert.ok(Math.abs(analytic - expected) < 1e-2, `en ${index}: ${analytic} vs ${expected}`);
    }
});

test("crossEntropy valida formas y objetivos", () => {
    assert.throws(() => crossEntropy(Tensor.from([1, 2], [2]), [0]), Error);
    assert.throws(() => crossEntropy(Tensor.from([1, 2], [1, 2]), [0, 1]), Error);
    assert.throws(() => crossEntropy(Tensor.from([1, 2], [1, 2]), [2]), RangeError);
    assert.throws(() => crossEntropy(Tensor.from([1, 2], [1, 2]), [0.5]), RangeError);
    assert.throws(() => crossEntropy(Tensor.zeros([0, 2]), []), Error);
    assert.throws(() => crossEntropy(Tensor.zeros([2, 0]), [0, 0]), Error);
});
