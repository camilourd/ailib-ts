import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import {
    coefficientOfDetermination,
    correlation,
    directionalAccuracy,
    meanAbsoluteError,
    meanError,
    rootMeanSquaredError,
} from "../../src/sets/metrics.ts";

test("MAE y RMSE coinciden con el cálculo a mano", () => {
    const prediction = Tensor.from([2, -3, 1], [3]);
    const target = Tensor.from([1, -1, 2], [3]);
    assert.ok(Math.abs(meanAbsoluteError(prediction, target) - 4 / 3) < 1e-6);
    assert.ok(Math.abs(rootMeanSquaredError(prediction, target) - Math.SQRT2) < 1e-6);
});

test("el acierto direccional cuenta los signos", () => {
    const prediction = Tensor.from([2, -3, 1, 0], [4]);
    const target = Tensor.from([1, -1, 2, 0], [4]);
    assert.equal(directionalAccuracy(prediction, target), 1);
    const wrong = Tensor.from([-2, 3, -1, 0], [4]);
    assert.equal(directionalAccuracy(wrong, target), 0.25);
});

test("las métricas validan formas y tamaño", () => {
    assert.throws(() => meanAbsoluteError(Tensor.from([1], [1]), Tensor.from([1, 2], [2])), Error);
    assert.throws(() => rootMeanSquaredError(Tensor.zeros([0]), Tensor.zeros([0])), Error);
});

test("sesgo, R² y correlación", () => {
    assert.equal(meanError(Tensor.from([1, 2], [2]), Tensor.from([0, 0], [2])), 1.5);

    const perfect = Tensor.from([1, -1, 2], [3]);
    assert.equal(coefficientOfDetermination(perfect, perfect), 1);
    assert.equal(correlation(perfect, perfect), 1);

    const inverted = Tensor.from([-1, 1, -2], [3]);
    assert.equal(correlation(inverted, perfect), -1);

    const meanPrediction = Tensor.from([0, 0], [2]);
    assert.equal(coefficientOfDetermination(meanPrediction, Tensor.from([1, -1], [2])), 0);

    const worse = Tensor.from([2, 2], [2]);
    assert.ok(coefficientOfDetermination(worse, Tensor.from([1, -1], [2])) < 0);

    assert.equal(correlation(Tensor.from([1, 1], [2]), Tensor.from([1, -1], [2])), 0);
});
