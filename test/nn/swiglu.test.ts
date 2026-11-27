import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { SwiGLU } from "../../src/nn/swiglu.ts";

test("SwiGLU: forma y gradiente", () => {
    const layer = new SwiGLU(3, 5, { seed: 1 });
    const input = Tensor.from([1, -2, 3, 0.5, 1.5, -0.5], [2, 3], { requiresGrad: true });
    const output = layer.forward(input);
    assert.deepEqual(output.shape, [2, 3]);
    sum(output).backward();
    assert.notEqual(input.grad, null);
    for (const parameter of layer.parameters()) assert.notEqual(parameter.grad, null);
    assert.equal(layer.parameters().length, 6);
});

test("SwiGLU: gradiente numérico de la entrada", () => {
    const layer = new SwiGLU(2, 3, { seed: 2 });
    const input = Tensor.from([0.5, -1, 1.5, 2], [2, 2], { requiresGrad: true });
    const loss = () => sum(layer.forward(input)).get();
    sum(layer.forward(input)).backward();
    for (let index = 0; index < input.size; index++) {
        const original = input.data[index] as number;
        input.data[index] = original + 1e-2;
        const plus = loss();
        input.data[index] = original - 1e-2;
        const minus = loss();
        input.data[index] = original;
        const numeric = (plus - minus) / 0.02;
        const analytic = input.grad?.data[index] ?? 0;
        assert.ok(Math.abs(analytic - numeric) < 1e-2, `${index}: ${analytic} vs ${numeric}`);
    }
});
