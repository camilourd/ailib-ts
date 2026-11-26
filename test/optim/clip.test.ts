import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { clipGradients } from "../../src/optim/clip.ts";

test("clipGradients recorta la norma global", () => {
    const first = Tensor.from([1], [1], { requiresGrad: true });
    const second = Tensor.from([1], [1], { requiresGrad: true });
    first.grad = Tensor.from([3], [1]);
    second.grad = Tensor.from([4], [1]);
    const norm = clipGradients([first, second], 1);
    assert.equal(norm, 5);
    assert.ok(Math.abs((first.grad?.data[0] as number) - 0.6) < 1e-6);
    assert.ok(Math.abs((second.grad?.data[0] as number) - 0.8) < 1e-6);
});

test("clipGradients no toca una norma pequeña y valida el límite", () => {
    const tensor = Tensor.from([1], [1], { requiresGrad: true });
    tensor.grad = Tensor.from([0.5], [1]);
    assert.equal(clipGradients([tensor], 1), 0.5);
    assert.equal(tensor.grad?.data[0], 0.5);
    assert.throws(() => clipGradients([tensor], 0), RangeError);
});
