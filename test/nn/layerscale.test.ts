import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { LayerScale } from "../../src/nn/layerscale.ts";

test("LayerScale escala por dimensión y propaga gradiente", () => {
    const layer = new LayerScale(3);
    const input = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const output = layer.forward(input);
    assert.deepEqual([...output.data], [1, 2, 3, 4, 5, 6]);
    sum(output).backward();
    assert.deepEqual([...layer.gamma.grad?.data ?? []], [5, 7, 9]);
    assert.notEqual(input.grad, null);
    assert.throws(() => new LayerScale(0), Error);
    assert.throws(() => layer.forward(Tensor.zeros([2, 2])), Error);
});
