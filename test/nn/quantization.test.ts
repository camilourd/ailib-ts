import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import {
    dequantizeInt8,
    floatBytes,
    quantizeInt8,
    quantizedBytes,
} from "../../src/nn/quantization.ts";

test("INT8: ida y vuelta con error acotado", () => {
    const tensor = Tensor.from([-1, -0.5, 0, 0.5, 1], [5]);
    const quantized = quantizeInt8(tensor);
    const restored = dequantizeInt8(quantized);
    for (let index = 0; index < tensor.size; index++) {
        assert.ok(
            Math.abs((restored.data[index] as number) - (tensor.data[index] as number)) < 0.01,
        );
    }
    assert.equal(quantized.scales.length, 1);
    assert.equal(quantizedBytes(quantized), 5 + 4);
    assert.equal(floatBytes(tensor), 20);
});

test("INT8 por canal usa una escala por columna", () => {
    const tensor = Tensor.from([1, 100, 2, 200], [2, 2]);
    const quantized = quantizeInt8(tensor, { perChannel: true });
    assert.equal(quantized.scales.length, 2);
    const restored = dequantizeInt8(quantized);
    assert.ok(Math.abs((restored.data[1] as number) - 100) < 2);
    assert.ok(Math.abs((restored.data[3] as number) - 200) < 2);
    const scalar = quantizeInt8(tensor, { perChannel: false });
    assert.equal(scalar.scales.length, 1);
});
