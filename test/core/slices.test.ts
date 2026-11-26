import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { concatColumns, sliceColumns, sliceRows, sum } from "../../src/core/tensor-ops.ts";

test("sliceColumns recorta y reparte el gradiente", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const slice = sliceColumns(tensor, 1, 3);
    assert.deepEqual([...slice.data], [2, 3, 5, 6]);
    sum(slice).backward();
    assert.deepEqual([...(tensor.grad?.data ?? [])], [0, 1, 1, 0, 1, 1]);
});

test("concatColumns junta y reparte el gradiente", () => {
    const first = Tensor.from([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const second = Tensor.from([5, 6, 7, 8], [2, 2], { requiresGrad: true });
    const joined = concatColumns([first, second]);
    assert.deepEqual([...joined.data], [1, 2, 5, 6, 3, 4, 7, 8]);
    sum(joined).backward();
    assert.deepEqual([...(first.grad?.data ?? [])], [1, 1, 1, 1]);
    assert.deepEqual([...(second.grad?.data ?? [])], [1, 1, 1, 1]);
});

test("sliceRows recorta y reparte el gradiente", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [3, 2], { requiresGrad: true });
    const slice = sliceRows(tensor, 1, 3);
    assert.deepEqual([...slice.data], [3, 4, 5, 6]);
    sum(slice).backward();
    assert.deepEqual([...(tensor.grad?.data ?? [])], [0, 0, 1, 1, 1, 1]);
});

test("sliceColumns y concatColumns validan formas", () => {
    assert.throws(() => sliceColumns(Tensor.zeros([2, 2]), 1, 3), RangeError);
    assert.throws(() => sliceRows(Tensor.zeros([2, 2]), 1, 3), RangeError);
    assert.throws(() => concatColumns([]), Error);
    assert.throws(() => concatColumns([Tensor.zeros([2, 2]), Tensor.zeros([3, 2])]), Error);
});
