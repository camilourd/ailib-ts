import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { add, matmul, mean, mul, sub, sum } from "../../src/core/tensor-ops.ts";
import { sigmoid, sigmoidTensor } from "../../src/nn/sigmoid.ts";

const tracked = (values: number[], shape?: readonly number[]): Tensor =>
    Tensor.from(values, shape ?? [values.length], { requiresGrad: true });

const gradientOf = (tensor: Tensor): number[] => [...(tensor.grad?.data ?? [])];

function numericGradient(f: (input: Tensor) => Tensor, input: Tensor, epsilon = 1e-2): Float32Array {
    const original = input.data.slice();
    const result = new Float32Array(input.size);
    for (let index = 0; index < input.size; index++) {
        input.data[index] = (original[index] as number) + epsilon;
        const plus = f(input).data.reduce((total, value) => total + value, 0);
        input.data[index] = (original[index] as number) - epsilon;
        const minus = f(input).data.reduce((total, value) => total + value, 0);
        input.data[index] = original[index] as number;
        result[index] = (plus - minus) / (2 * epsilon);
    }
    return result;
}

function numericGradientOf(forward: () => Tensor, parameter: Tensor, epsilon = 1e-2): Float32Array {
    const original = parameter.data.slice();
    const result = new Float32Array(parameter.size);
    for (let index = 0; index < parameter.size; index++) {
        parameter.data[index] = (original[index] as number) + epsilon;
        const plus = forward().get();
        parameter.data[index] = (original[index] as number) - epsilon;
        const minus = forward().get();
        parameter.data[index] = original[index] as number;
        result[index] = (plus - minus) / (2 * epsilon);
    }
    return result;
}

test("add reparte el gradiente entre sus entradas", () => {
    const left = tracked([1, 2, 3]);
    const right = tracked([4, 5, 6]);
    const output = add(left, right);
    output.backward();
    assert.deepEqual(gradientOf(left), [1, 1, 1]);
    assert.deepEqual(gradientOf(right), [1, 1, 1]);
});

test("mul usa el valor de la otra entrada", () => {
    const left = tracked([1, 2, 3]);
    const right = tracked([4, 5, 6]);
    const output = mul(left, right);
    output.backward();
    assert.deepEqual(gradientOf(left), [4, 5, 6]);
    assert.deepEqual(gradientOf(right), [1, 2, 3]);
});

test("el broadcasting acumula el gradiente en el operando expandido", () => {
    const matrix = tracked([1, 2, 3, 4, 5, 6], [2, 3]);
    const bias = tracked([10, 20, 30]);
    const output = add(matrix, bias);
    output.backward();
    assert.deepEqual(gradientOf(matrix), [1, 1, 1, 1, 1, 1]);
    assert.deepEqual(gradientOf(bias), [2, 2, 2]);
});

test("una cadena aplica la regla de la cadena", () => {
    const input = tracked([1, 2, 3]);
    const output = mul(add(input, input), input);
    output.backward();
    assert.deepEqual(gradientOf(input), [4, 8, 12]);
});

test("sin requiresGrad no se construye grafo", () => {
    const left = Tensor.from([1, 2, 3]);
    const right = Tensor.from([4, 5, 6]);
    const output = add(left, right);
    assert.equal(output.requiresGrad, false);
    output.backward();
    assert.equal(left.grad, null);
});

test("zeroGrad limpia el gradiente", () => {
    const input = tracked([1, 2, 3]);
    mul(input, input).backward();
    assert.notEqual(input.grad, null);
    input.zeroGrad();
    assert.equal(input.grad, null);
});

test("la derivada analítica coincide con la numérica", () => {
    const input = tracked([0.5, 1, 1.5]);
    const analytic = mul(input, input);
    analytic.backward();
    const numeric = numericGradient((tensor) => mul(tensor, tensor), input);
    for (let index = 0; index < input.size; index++) {
        const expected = numeric[index] as number;
        const actual = (input.grad?.data[index] ?? 0) as number;
        assert.ok(Math.abs(actual - expected) < 1e-2, `en ${index}: ${actual} vs ${expected}`);
    }
});

test("sum y mean reparten el gradiente que toca", () => {
    const input = tracked([1, 2, 3, 4], [2, 2]);

    sum(input).backward();
    assert.deepEqual(gradientOf(input), [1, 1, 1, 1]);

    input.zeroGrad();
    mean(input).backward();
    assert.deepEqual(gradientOf(input), [0.25, 0.25, 0.25, 0.25]);

    input.zeroGrad();
    sum(input, 0).backward();
    assert.deepEqual(gradientOf(input), [1, 1, 1, 1]);

    input.zeroGrad();
    sum(input, 1).backward();
    assert.deepEqual(gradientOf(input), [1, 1, 1, 1]);
});

test("sigmoid: el gradiente es σ(1-σ)", () => {
    const input = tracked([-1, 0, 1]);
    sigmoidTensor(input).backward();
    [-1, 0, 1].forEach((value, index) => {
        const expected = sigmoid(value) * (1 - sigmoid(value));
        const actual = (input.grad?.data[index] ?? 0) as number;
        assert.ok(Math.abs(actual - expected) < 1e-6, `en ${index}: ${actual} vs ${expected}`);
    });
});

test("matmul: el gradiente coincide con la derivada numérica", () => {
    const left = tracked([1, 2, 3, 4, 5, 6], [2, 3]);
    const right = tracked([7, 8, 9, 10, 11, 12], [3, 2]);
    sum(matmul(left, right)).backward();

    const numericLeft = numericGradient((tensor) => sum(matmul(tensor, right)), left);
    for (let index = 0; index < left.size; index++) {
        const analytic = (left.grad?.data[index] ?? 0) as number;
        const expected = numericLeft[index] as number;
        assert.ok(Math.abs(analytic - expected) < 1e-2, `left ${index}: ${analytic} vs ${expected}`);
    }

    const numericRight = numericGradient((tensor) => sum(matmul(left, tensor)), right);
    for (let index = 0; index < right.size; index++) {
        const analytic = (right.grad?.data[index] ?? 0) as number;
        const expected = numericRight[index] as number;
        assert.ok(Math.abs(analytic - expected) < 1e-2, `right ${index}: ${analytic} vs ${expected}`);
    }
});

test("gradcheck de un MLP de dos capas", () => {
    const input = Tensor.from([0.5, -0.3, 0.8, 0.1], [2, 2]);
    const target = Tensor.from([1, 0], [2, 1]);
    const weights1 = tracked([0.2, -0.1, 0.3, 0.15, -0.25, 0.05], [2, 3]);
    const bias1 = tracked([0.01, -0.02, 0.03], [3]);
    const weights2 = tracked([0.4, -0.3, 0.2], [3, 1]);
    const bias2 = tracked([-0.05], [1]);

    const forward = (): Tensor => {
        const hidden = sigmoidTensor(add(matmul(input, weights1), bias1));
        const prediction = sigmoidTensor(add(matmul(hidden, weights2), bias2));
        const difference = sub(prediction, target);
        return mean(mul(difference, difference));
    };

    forward().backward();

    for (const parameter of [weights1, bias1, weights2, bias2]) {
        const numeric = numericGradientOf(forward, parameter);
        for (let index = 0; index < parameter.size; index++) {
            const analytic = (parameter.grad?.data[index] ?? 0) as number;
            const expected = numeric[index] as number;
            assert.ok(Math.abs(analytic - expected) < 2e-2, `en ${index}: ${analytic} vs ${expected}`);
        }
    }
});
