import assert from "node:assert/strict";
import { test } from "node:test";

import { createRng } from "../../src/core/random/index.ts";
import { sampleFromValues } from "../../src/nn/sampling.ts";

test("sampleFromValues es reproducible y respeta la temperatura", () => {
    const values = [0, 0, 0, 0];
    const first = sampleFromValues(values, 1, createRng(3));
    const second = sampleFromValues(values, 1, createRng(3));
    assert.equal(first, second);
    assert.ok(first >= 0 && first < 4);
    assert.equal(sampleFromValues([10, 0], 0.01, createRng(1)), 0);
});

test("sampleFromValues valida temperatura y fila", () => {
    assert.throws(() => sampleFromValues([1], 0, createRng(1)), RangeError);
    assert.throws(() => sampleFromValues([], 1, createRng(1)), Error);
});

test("top-k y top-p recortan la distribución", () => {
    assert.equal(sampleFromValues([0, 10], { temperature: 1, topK: 1 }, createRng(1)), 1);
    assert.equal(sampleFromValues([10, 0], { temperature: 1, topP: 0.5 }, createRng(1)), 0);
    assert.throws(
        () => sampleFromValues([1, 2], { temperature: 1, topK: 0 }, createRng(1)),
        RangeError,
    );
    assert.throws(
        () => sampleFromValues([1, 2], { temperature: 1, topP: 0 }, createRng(1)),
        RangeError,
    );
    assert.throws(
        () => sampleFromValues([1, 2], { temperature: 1, topP: 1.1 }, createRng(1)),
        RangeError,
    );
});
