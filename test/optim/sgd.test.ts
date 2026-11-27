import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { mul } from "../../src/core/tensor-ops.ts";
import { SGD } from "../../src/optim/sgd.ts";

test("SGD baja una cuadrática hasta el mínimo", () => {
    const x = Tensor.from([4], [1], { requiresGrad: true });
    const optimizer = new SGD([x], 0.1);
    for (let step = 0; step < 100; step++) {
        optimizer.zeroGrad();
        mul(x, x).backward();
        optimizer.step();
    }
    assert.ok(Math.abs(x.get(0)) < 1e-6, `x = ${x.get(0)}`);
});

test("SGD actualiza cada parámetro con su gradiente", () => {
    const a = Tensor.from([1, 2], [2], { requiresGrad: true });
    const b = Tensor.from([3], [1], { requiresGrad: true });
    a.grad = Tensor.from([1, 1], [2]);
    b.grad = Tensor.from([2], [1]);
    new SGD([a, b], 0.5).step();
    assert.deepEqual([...a.data], [0.5, 1.5]);
    assert.deepEqual([...b.data], [2]);
});

test("zeroGrad limpia todos los gradientes", () => {
    const a = Tensor.from([1], [1], { requiresGrad: true });
    const b = Tensor.from([2], [1], { requiresGrad: true });
    a.grad = Tensor.from([1], [1]);
    b.grad = Tensor.from([1], [1]);
    const optimizer = new SGD([a, b], 0.1);
    optimizer.zeroGrad();
    assert.equal(a.grad, null);
    assert.equal(b.grad, null);
});

test("un parámetro sin gradiente no se toca", () => {
    const withoutGrad = Tensor.from([5], [1]);
    const withGrad = Tensor.from([5], [1], { requiresGrad: true });
    new SGD([withoutGrad, withGrad], 0.1).step();
    assert.equal(withoutGrad.get(0), 5);
    assert.equal(withGrad.get(0), 5);
});

test("una tasa de aprendizaje inválida falla", () => {
    assert.throws(() => new SGD([], 0), RangeError);
    assert.throws(() => new SGD([], -0.1), RangeError);
    assert.throws(() => new SGD([], Number.NaN), RangeError);
});

test("la tasa de aprendizaje se puede cambiar con validación", () => {
    const optimizer = new SGD([], 0.1);
    optimizer.learningRate = 0.2;
    assert.equal(optimizer.learningRate, 0.2);
    assert.throws(() => {
        optimizer.learningRate = 0;
    }, RangeError);
});

test("weight decay encoge el parámetro además del gradiente", () => {
    const parameter = Tensor.from([2], [1], { requiresGrad: true });
    parameter.grad = Tensor.from([0], [1]);
    new SGD([parameter], 0.1, { weightDecay: 0.5 }).step();
    assert.ok(Math.abs(parameter.get(0) - 1.9) < 1e-6);
    assert.throws(() => new SGD([], 0.1, { weightDecay: -1 }), RangeError);
});
