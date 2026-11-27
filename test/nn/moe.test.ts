import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { MoEFFN } from "../../src/nn/moe.ts";

test("MoE top-1 usa un experto por fila", () => {
    const layer = new MoEFFN(4, 8, { experts: 4, topK: 1, seed: 1 });
    const input = Tensor.from(new Array(24).fill(0.5), [6, 4], { requiresGrad: true });
    const output = layer.forward(input);
    assert.deepEqual(output.shape, [6, 4]);
    const usage = layer.usage();
    assert.equal(usage.reduce((total, value) => total + value, 0), 6);
    assert.ok(usage.some((value) => value > 0));
});

test("MoE propaga gradiente a router y expertos usados", () => {
    const layer = new MoEFFN(4, 8, { experts: 3, topK: 1, seed: 2 });
    const input = Tensor.from(new Array(16).fill(0.25), [4, 4], { requiresGrad: true });
    sum(layer.forward(input)).backward();
    assert.notEqual(input.grad, null);
    for (const parameter of layer.parameters()) assert.notEqual(parameter.grad, null);
    assert.throws(() => new MoEFFN(4, 8, { experts: 2, topK: 3 }), Error);
});
