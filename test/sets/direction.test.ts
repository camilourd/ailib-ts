import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { confusionMatrix, directionReport, errorsByDirection } from "../../src/sets/direction.ts";

test("la matriz de confusión excluye los ceros reales", () => {
    const prediction = Tensor.from([1, -1, 1, -1, 0], [5]);
    const target = Tensor.from([1, 1, -1, -1, 0], [5]);
    assert.deepEqual(confusionMatrix(prediction, target), {
        truePositive: 1,
        falsePositive: 1,
        trueNegative: 1,
        falseNegative: 1,
        ties: 1,
        total: 5,
    });
});

test("el informe de dirección coincide con el cálculo a mano", () => {
    const prediction = Tensor.from([1, -1, 1, -1, 0], [5]);
    const target = Tensor.from([1, 1, -1, -1, 0], [5]);
    const report = directionReport(prediction, target);
    assert.equal(report.evaluated, 4);
    assert.ok(Math.abs(report.coverage - 0.8) < 1e-12);
    assert.equal(report.accuracy, 0.5);
    assert.equal(report.precision, 0.5);
    assert.equal(report.recall, 0.5);
    assert.equal(report.specificity, 0.5);
    assert.equal(report.f1, 0.5);
    assert.equal(report.balancedAccuracy, 0.5);
    assert.equal(report.mcc, 0);
    assert.equal(report.predictedUpRate, 0.5);
    assert.equal(report.actualUpRate, 0.5);
});

test("todo bien y todo mal llevan las métricas a los extremos", () => {
    const good = directionReport(
        Tensor.from([1, -1, 1], [3]),
        Tensor.from([1, -1, 1], [3]),
    );
    assert.equal(good.accuracy, 1);
    assert.equal(good.precision, 1);
    assert.equal(good.recall, 1);
    assert.equal(good.specificity, 1);
    assert.equal(good.f1, 1);
    assert.equal(good.balancedAccuracy, 1);
    assert.equal(good.mcc, 1);

    const bad = directionReport(
        Tensor.from([-1, 1, -1], [3]),
        Tensor.from([1, -1, 1], [3]),
    );
    assert.equal(bad.accuracy, 0);
    assert.equal(bad.precision, 0);
    assert.equal(bad.recall, 0);
    assert.equal(bad.specificity, 0);
    assert.equal(bad.f1, 0);
    assert.equal(bad.balancedAccuracy, 0);
    assert.equal(bad.mcc, -1);
});

test("sin positivos predichos o sin positivos reales no divide entre cero", () => {
    const noPredictedUp = directionReport(
        Tensor.from([-1, -1], [2]),
        Tensor.from([1, -1], [2]),
    );
    assert.equal(noPredictedUp.precision, 0);
    assert.equal(noPredictedUp.recall, 0);
    assert.equal(noPredictedUp.specificity, 1);
    assert.equal(noPredictedUp.accuracy, 0.5);

    const noActualUp = directionReport(
        Tensor.from([1, -1], [2]),
        Tensor.from([-1, -1], [2]),
    );
    assert.equal(noActualUp.precision, 0);
    assert.equal(noActualUp.recall, 0);
    assert.equal(noActualUp.specificity, 0.5);
    assert.equal(noActualUp.accuracy, 0.5);
});

test("un informe sin días direccionales queda a cero", () => {
    const report = directionReport(Tensor.from([1, -1], [2]), Tensor.from([0, 0], [2]));
    assert.equal(report.evaluated, 0);
    assert.equal(report.coverage, 0);
    assert.equal(report.accuracy, 0);
    assert.equal(report.precision, 0);
    assert.equal(report.recall, 0);
    assert.equal(report.specificity, 0);
    assert.equal(report.mcc, 0);
});

test("el error por dirección separa los días que suben y los que bajan", () => {
    const errors = errorsByDirection(
        Tensor.from([0, 0], [2]),
        Tensor.from([1, -1], [2]),
    );
    assert.equal(errors.up, 1);
    assert.equal(errors.down, 1);
    assert.equal(errors.upCount, 1);
    assert.equal(errors.downCount, 1);
});

test("la dirección valida formas y tamaño", () => {
    assert.throws(
        () => confusionMatrix(Tensor.from([1], [1]), Tensor.from([1, 2], [2])),
        Error,
    );
    assert.throws(() => directionReport(Tensor.zeros([0]), Tensor.zeros([0])), Error);
});
