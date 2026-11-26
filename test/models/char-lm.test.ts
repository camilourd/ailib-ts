import assert from "node:assert/strict";
import { test } from "node:test";

import { sum } from "../../src/core/tensor-ops.ts";
import { CharLM } from "../../src/models/char-lm.ts";

test("CharLM: formas de logits y estados", () => {
    const model = new CharLM(4, { embeddingSize: 4, hiddenSize: 4, seed: 1 });
    const { logits, states } = model.forward([0, 1, 2]);
    assert.equal(logits.length, 3);
    for (const logit of logits) assert.deepEqual(logit.shape, [1, 4]);
    for (const state of states) assert.deepEqual(state.shape, [1, 4]);
    assert.equal(model.parameters().length, 6);
});

test("CharLM: la pérdida es escalar y el gradiente llega", () => {
    const model = new CharLM(4, { embeddingSize: 4, hiddenSize: 4, seed: 1 });
    const loss = model.loss([0, 1, 2, 3]);
    assert.equal(loss.rank, 0);
    assert.ok(Number.isFinite(loss.get()));
    sum(loss).backward();
    for (const parameter of model.parameters()) {
        assert.notEqual(parameter.grad, null);
    }
});

test("CharLM: genera bytes y es reproducible por semilla", () => {
    const model = new CharLM(4, { embeddingSize: 4, hiddenSize: 4, seed: 1 });
    const first = model.generate([0], 5, { temperature: 1, seed: 2 });
    const second = model.generate([0], 5, { temperature: 1, seed: 2 });
    assert.equal(first.length, 5);
    assert.deepEqual(first, second);
    for (const token of first) assert.ok(token >= 0 && token < 4);
});

test("CharLM: valida tokens, pérdida y generación", () => {
    const model = new CharLM(4);
    assert.throws(() => model.forward([]), Error);
    assert.throws(() => model.loss([0]), Error);
    assert.throws(() => model.generate([], 1), Error);
    assert.throws(() => model.generate([0], -1), RangeError);
    assert.throws(() => model.generate([0], 1, { temperature: 0 }), RangeError);
});
