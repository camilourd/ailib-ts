import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import type { AttentionCache } from "../../src/nn/kv-cache.ts";
import { MultiHeadAttention } from "../../src/nn/multihead.ts";

test("multi-head: formas, cabezas y máscara", () => {
    const attention = new MultiHeadAttention(4, 2, { seed: 1 });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
        { requiresGrad: true },
    );
    const { output, weights } = attention.forward(input);
    assert.deepEqual(output.shape, [3, 4]);
    assert.equal(weights.length, 2);
    for (const weight of weights) assert.deepEqual(weight.shape, [3, 3]);
    for (const weight of weights) {
        assert.equal(weight.data[1], 0);
        assert.equal(weight.data[2], 0);
        assert.equal(weight.data[5], 0);
    }
    assert.equal(attention.parameters().length, 8);
});

test("multi-head: el gradiente llega a la entrada y a las proyecciones", () => {
    const attention = new MultiHeadAttention(4, 2, { seed: 1 });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
        { requiresGrad: true },
    );
    sum(attention.forward(input).output).backward();
    for (const parameter of attention.parameters()) assert.notEqual(parameter.grad, null);
    assert.notEqual(input.grad, null);
});

test("multi-head: valida dimensiones y cabezas", () => {
    assert.throws(() => new MultiHeadAttention(4, 3), Error);
    assert.throws(() => new MultiHeadAttention(4, 0), Error);
    const attention = new MultiHeadAttention(4, 2);
    assert.throws(() => attention.forward(Tensor.zeros([2, 3])), Error);
});

test("multi-head: forwardStep coincide con forward", () => {
    const attention = new MultiHeadAttention(4, 2, { seed: 1, rope: true });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
    );
    const baseline = attention.forward(input).output;
    const cache: AttentionCache = {
        keys: [Tensor.zeros([0, 2]), Tensor.zeros([0, 2])],
        values: [Tensor.zeros([0, 2]), Tensor.zeros([0, 2])],
    };
    for (let position = 0; position < 3; position++) {
        const row = new Tensor(
            input.data.subarray(position * 4, (position + 1) * 4),
            [1, 4],
        );
        const cached = attention.forwardStep(row, cache, position);
        for (let column = 0; column < 4; column++) {
            assert.ok(
                Math.abs((cached.data[column] as number) - (baseline.data[position * 4 + column] as number)) < 1e-5,
                `posición ${position}, columna ${column}`,
            );
        }
    }
});
