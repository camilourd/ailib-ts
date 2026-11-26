import assert from "node:assert/strict";
import { test } from "node:test";

import { sigmoid, sigmoidDerivative } from "../../src/nn/sigmoid.ts";

const POINTS = [-8, -3, -1, 0, 0.5, 2, 5, 9];

test("σ lleva los reales a (0, 1) y es estrictamente creciente", () => {
    let previous = -Infinity;
    for (const x of POINTS) {
        const value = sigmoid(x);
        assert.ok(value > 0 && value < 1, `σ(${x}) = ${value} fuera de (0,1)`);
        assert.ok(value > previous, `σ no es creciente en x = ${x}`);
        previous = value;
    }
});

test("simetría: σ(-x) = 1 - σ(x)", () => {
    for (const x of POINTS) {
        assert.ok(Math.abs(sigmoid(-x) + sigmoid(x) - 1) < 1e-12, `falla en x = ${x}`);
    }
});

test("la derivada analítica coincide con la numérica", () => {
    const h = 1e-6;
    for (const x of POINTS) {
        const numeric = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h);
        const analytic = sigmoidDerivative(x);
        assert.ok(
            Math.abs(numeric - analytic) < 1e-8,
            `en x = ${x}: numérica ${numeric}, analítica ${analytic}`,
        );
    }
});

test("saturación: en las colas la derivada es casi nula", () => {
    for (const x of [-50, 50]) {
        assert.ok(sigmoidDerivative(x) < 1e-20, `σ'(${x}) no está saturada`);
    }
});

test("estabilidad: no hay NaN ni Infinity en los extremos", () => {
    for (const x of [-1000, -100, 100, 1000]) {
        assert.ok(Number.isFinite(sigmoid(x)), `σ(${x}) no es finito`);
        assert.ok(Number.isFinite(sigmoidDerivative(x)), `σ'(${x}) no es finito`);
    }
});
