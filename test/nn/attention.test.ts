import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { scaledDotProductAttention } from "../../src/nn/attention.ts";

test("atención: ejemplo a mano sin máscara", () => {
    const queries = Tensor.from([1, 1], [2, 1]);
    const keys = Tensor.from([1, 0], [2, 1]);
    const values = Tensor.from([2, 4], [2, 1]);
    const { output, weights } = scaledDotProductAttention(queries, keys, values, {
        causal: false,
    });
    const first = Math.exp(1) / (Math.exp(1) + 1);
    assert.ok(Math.abs((weights.data[0] as number) - first) < 1e-6);
    assert.ok(Math.abs((weights.data[1] as number) - (1 - first)) < 1e-6);
    assert.ok(Math.abs((output.data[0] as number) - (2 * first + 4 * (1 - first))) < 1e-5);
});

test("la máscara causal deja ceros por encima de la diagonal", () => {
    const queries = Tensor.from([1, 1, 1], [3, 1]);
    const keys = Tensor.from([1, 1, 1], [3, 1]);
    const values = Tensor.from([1, 2, 3], [3, 1]);
    const { output, weights } = scaledDotProductAttention(queries, keys, values);
    assert.equal(weights.data[1], 0);
    assert.equal(weights.data[2], 0);
    assert.equal(weights.data[5], 0);
    assert.ok(Math.abs((output.data[0] as number) - 1) < 1e-6);
    assert.ok(Math.abs((output.data[1] as number) - 1.5) < 1e-6);
    assert.ok(Math.abs((output.data[2] as number) - 2) < 1e-6);
});

test("la atención valida formas", () => {
    const queries = Tensor.from([1, 2], [1, 2]);
    assert.throws(
        () => scaledDotProductAttention(queries, Tensor.from([1, 2, 3], [1, 3]), Tensor.zeros([1, 2])),
        Error,
    );
    assert.throws(
        () => scaledDotProductAttention(queries, Tensor.zeros([2, 2]), Tensor.zeros([1, 2])),
        Error,
    );
});

test("el gradiente de Q, K y V coincide con diferencias finitas", () => {
    const q = Tensor.from([0.5, 0.2, -0.1, 0.7], [2, 2], { requiresGrad: true });
    const k = Tensor.from([0.3, -0.4, 0.6, 0.1], [2, 2], { requiresGrad: true });
    const v = Tensor.from([1, -2, 3, 0.5], [2, 2], { requiresGrad: true });
    const loss = () => sum(scaledDotProductAttention(q, k, v).output).get();
    sum(scaledDotProductAttention(q, k, v).output).backward();

    for (const [tensor, name] of [
        [q, "Q"],
        [k, "K"],
        [v, "V"],
    ] as const) {
        for (let index = 0; index < tensor.size; index++) {
            const original = tensor.data[index] as number;
            tensor.data[index] = original + 1e-2;
            const plus = loss();
            tensor.data[index] = original - 1e-2;
            const minus = loss();
            tensor.data[index] = original;
            const numeric = (plus - minus) / 0.02;
            const analytic = tensor.grad?.data[index] ?? 0;
            assert.ok(
                Math.abs(analytic - numeric) < 1e-2,
                `${name}[${index}]: ${analytic} vs ${numeric}`,
            );
        }
    }
});
