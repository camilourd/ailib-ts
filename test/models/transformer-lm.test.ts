import assert from "node:assert/strict";
import { test } from "node:test";

import { TransformerLM } from "../../src/models/transformer-lm.ts";

test("TransformerLM: logits por token, pérdida y gradiente", () => {
    const model = new TransformerLM(8, {
        modelSize: 8,
        heads: 2,
        blocks: 1,
        hiddenSize: 16,
        seed: 1,
    });
    const logits = model.forward([0, 1, 2]);
    assert.deepEqual(logits.shape, [3, 8]);
    const loss = model.loss([0, 1, 2, 3]);
    assert.equal(loss.rank, 0);
    assert.ok(Number.isFinite(loss.get()));
    loss.backward();
    for (const parameter of model.parameters()) assert.notEqual(parameter.grad, null);
});

test("TransformerLM: generación reproducible", () => {
    const model = new TransformerLM(8, {
        modelSize: 8,
        heads: 2,
        blocks: 1,
        hiddenSize: 16,
        seed: 1,
    });
    const first = model.generate([0], 4, { temperature: 1, seed: 2 });
    const second = model.generate([0], 4, { temperature: 1, seed: 2 });
    assert.deepEqual(first, second);
    assert.equal(first.length, 4);
    for (const token of first) assert.ok(token >= 0 && token < 8);
});

test("TransformerLM: generación con KV cache coincide", () => {
    const model = new TransformerLM(8, {
        modelSize: 8,
        heads: 2,
        blocks: 1,
        hiddenSize: 16,
        seed: 1,
    });
    const normal = model.generate([0, 1], 4, { temperature: 0.8, seed: 2 });
    const cached = model.generate([0, 1], 4, {
        temperature: 0.8,
        seed: 2,
        useCache: true,
    });
    assert.deepEqual(cached, normal);
});

test("TransformerLM: valida opciones y entradas", () => {
    assert.throws(() => new TransformerLM(8, { blocks: 0 }), Error);
    assert.throws(() => new TransformerLM(8, { modelSize: 8, heads: 3 }), Error);
    const model = new TransformerLM(8);
    assert.throws(() => model.forward([]), Error);
    assert.throws(() => model.loss([0]), Error);
    assert.throws(() => model.generate([], 1), Error);
    assert.throws(() => model.generate([0], 1, { temperature: 0 }), RangeError);
});

test("TransformerLM: copia parámetros entre instancias", () => {
    const source = new TransformerLM(8, {
        modelSize: 8,
        heads: 2,
        blocks: 1,
        hiddenSize: 16,
        seed: 1,
    });
    const target = new TransformerLM(8, {
        modelSize: 8,
        heads: 2,
        blocks: 1,
        hiddenSize: 16,
        seed: 2,
    });
    target.loadParameters(source.namedParameters());
    assert.deepEqual([...target.forward([0, 1]).data], [...source.forward([0, 1]).data]);
    assert.throws(() => target.loadParameters({}), Error);
});

test("TransformerLM: configuración de 1–10M parámetros", () => {
    const model = new TransformerLM(512, {
        modelSize: 160,
        heads: 8,
        blocks: 4,
        hiddenSize: 640,
        seed: 1,
    });
    assert.deepEqual(model.architecture, {
        modelSize: 160,
        heads: 8,
        blocks: 4,
        hiddenSize: 640,
        vocabularySize: 512,
    });
    assert.equal(model.parameterCount(), 1_401_792);
    assert.ok(model.parameterCount() >= 1_000_000);
    assert.ok(model.parameterCount() <= 10_000_000);
    assert.equal(Object.keys(model.namedParameters()).length, model.parameters().length);
});
