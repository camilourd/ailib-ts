import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { RMSNorm } from "../../src/nn/rmsnorm.ts";

test("RMSNorm normaliza el RMS por fila", () => {
    const layer = new RMSNorm(3);
    const input = Tensor.from([1, 2, 3, 10, 20, 30], [2, 3]);
    const output = layer.forward(input);
    for (let row = 0; row < 2; row++) {
        let squares = 0;
        for (let column = 0; column < 3; column++) {
            const value = output.get(row, column);
            squares += value * value;
        }
        const rms = Math.sqrt(squares / 3);
        assert.ok(Math.abs(rms - 1) < 1e-4, `rms ${rms}`);
    }
});

test("RMSNorm: gradiente de entrada y gamma", () => {
    const layer = new RMSNorm(3);
    const input = Tensor.from([1, 2, 4, 0.5, -1, 2], [2, 3], { requiresGrad: true });
    const loss = () => sum(layer.forward(input)).get();
    sum(layer.forward(input)).backward();

    for (const [tensor, name] of [
        [input, "entrada"],
        [layer.gamma, "gamma"],
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

test("RMSNorm valida forma y epsilon", () => {
    assert.throws(() => new RMSNorm(0), Error);
    assert.throws(() => new RMSNorm(3, { epsilon: 0 }), RangeError);
    assert.throws(() => new RMSNorm(3).forward(Tensor.zeros([2, 2])), Error);
});
