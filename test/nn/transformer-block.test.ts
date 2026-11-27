import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { TransformerBlock } from "../../src/nn/transformer-block.ts";

test("bloque transformer: formas, pesos y máscara", () => {
    const block = new TransformerBlock(4, 2, { hiddenSize: 8, seed: 1 });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
        { requiresGrad: true },
    );
    const { output, weights } = block.forward(input);
    assert.deepEqual(output.shape, [3, 4]);
    assert.equal(weights.length, 2);
    for (const weight of weights) assert.deepEqual(weight.shape, [3, 3]);
    for (const weight of weights) assert.equal(weight.data[1], 0);
    assert.equal(block.parameters().length, 16);
});

test("bloque transformer: el gradiente llega a todas las piezas", () => {
    const block = new TransformerBlock(4, 2, { hiddenSize: 8, seed: 1 });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
        { requiresGrad: true },
    );
    sum(block.forward(input).output).backward();
    for (const parameter of block.parameters()) assert.notEqual(parameter.grad, null);
    assert.notEqual(input.grad, null);
});

test("bloque transformer: valida cabezas y entrada", () => {
    assert.throws(() => new TransformerBlock(4, 3), Error);
    const block = new TransformerBlock(4, 2, { hiddenSize: 8 });
    assert.throws(() => block.forward(Tensor.zeros([2, 3])), Error);
});

test("bloque transformer: se pueden apagar atención o FFN", () => {
    const input = Tensor.zeros([3, 4]);
    const noAttention = new TransformerBlock(4, 2, { attention: false, hiddenSize: 8 });
    assert.equal(noAttention.parameters().length, 6);
    assert.equal(noAttention.forward(input).weights.length, 0);
    const noFfn = new TransformerBlock(4, 2, { ffn: false, hiddenSize: 8 });
    assert.equal(noFfn.parameters().length, 10);
    assert.equal(noFfn.forward(input).weights.length, 2);
});

test("bloque transformer moderno: RMSNorm + SwiGLU + paralelo", () => {
    const block = new TransformerBlock(4, 2, {
        hiddenSize: 8,
        seed: 3,
        normKind: "rms",
        ffnKind: "swiglu",
        parallel: true,
    });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
        { requiresGrad: true },
    );
    const output = block.forward(input).output;
    assert.deepEqual(output.shape, [3, 4]);
    sum(output).backward();
    for (const parameter of block.parameters()) assert.notEqual(parameter.grad, null);
    assert.notEqual(input.grad, null);
});
