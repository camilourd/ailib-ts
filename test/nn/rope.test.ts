import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { MultiHeadAttention } from "../../src/nn/multihead.ts";
import { rope } from "../../src/nn/rope.ts";

const dot = (first: Tensor, second: Tensor): number => {
    let total = 0;
    for (let index = 0; index < first.size; index++) {
        total += (first.data[index] as number) * (second.data[index] as number);
    }
    return total;
};

test("RoPE en la posición 0 no cambia y conserva la norma", () => {
    const vector = Tensor.from([1, 2], [1, 2]);
    assert.deepEqual([...rope(vector).data], [1, 2]);
    const other = Tensor.from([0.5, -1.5], [1, 2]);
    const rotated = rope(other, { offset: 3 });
    assert.ok(Math.abs(dot(other, other) - dot(rotated, rotated)) < 1e-5);
});

test("RoPE hace que q·k dependa de la distancia", () => {
    const query = Tensor.from([1, 0.5], [1, 2]);
    const key = Tensor.from([0.3, -0.7], [1, 2]);
    const first = dot(rope(query, { offset: 3 }), rope(key, { offset: 5 }));
    const second = dot(rope(query, { offset: 0 }), rope(key, { offset: 2 }));
    assert.ok(Math.abs(first - second) < 1e-5, `${first} vs ${second}`);
});

test("RoPE: gradiente de la entrada", () => {
    const tensor = Tensor.from([0.5, -1, 2, 0.3], [2, 2], { requiresGrad: true });
    const loss = () => sum(rope(tensor)).get();
    sum(rope(tensor)).backward();
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
            `índice ${index}: ${analytic} vs ${numeric}`,
        );
    }
});

test("multi-head con RoPE conserva formas y gradiente", () => {
    const attention = new MultiHeadAttention(4, 2, { seed: 1, rope: true });
    const input = Tensor.from(
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [3, 4],
        { requiresGrad: true },
    );
    const { output, weights } = attention.forward(input);
    assert.deepEqual(output.shape, [3, 4]);
    assert.equal(weights[0]?.data[1], 0);
    sum(output).backward();
    for (const parameter of attention.parameters()) assert.notEqual(parameter.grad, null);
    assert.notEqual(input.grad, null);
});

test("RoPE valida dimensión, base y desplazamiento", () => {
    assert.throws(() => rope(Tensor.from([1, 2, 3], [1, 3])), Error);
    assert.throws(() => rope(Tensor.from([1, 2], [1, 2]), { base: 0 }), RangeError);
    assert.throws(() => rope(Tensor.from([1, 2], [1, 2]), { offset: -1 }), RangeError);
    assert.throws(() => rope(Tensor.from([1, 2], [2])), Error);
});
