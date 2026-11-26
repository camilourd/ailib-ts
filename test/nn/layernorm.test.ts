import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { LayerNorm } from "../../src/nn/layernorm.ts";

test("LayerNorm normaliza cada fila", () => {
    const layer = new LayerNorm(3);
    const input = Tensor.from([1, 2, 3, 10, 20, 30], [2, 3]);
    const output = layer.forward(input);
    for (let row = 0; row < 2; row++) {
        let mean = 0;
        for (let column = 0; column < 3; column++) mean += output.get(row, column);
        mean /= 3;
        let variance = 0;
        for (let column = 0; column < 3; column++) {
            const delta = output.get(row, column) - mean;
            variance += delta * delta;
        }
        variance /= 3;
        assert.ok(Math.abs(mean) < 1e-5, `media ${mean}`);
        assert.ok(Math.abs(variance - 1) < 1e-4, `varianza ${variance}`);
    }
});

test("LayerNorm: gradiente de entrada, gamma y beta", () => {
    const layer = new LayerNorm(3);
    const input = Tensor.from([1, 2, 4, 0.5, -1, 2], [2, 3], { requiresGrad: true });
    const loss = () => sum(layer.forward(input)).get();
    sum(layer.forward(input)).backward();

    for (const [tensor, name] of [
        [input, "entrada"],
        [layer.gamma, "gamma"],
        [layer.beta, "beta"],
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

test("LayerNorm valida formas y epsilon", () => {
    assert.throws(() => new LayerNorm(0), Error);
    assert.throws(() => new LayerNorm(3, { epsilon: 0 }), RangeError);
    assert.throws(() => new LayerNorm(3).forward(Tensor.zeros([2, 2])), Error);
});
