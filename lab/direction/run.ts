// Teoría: docs/teoria/entrenamiento/direccion.md
// Experimento: matriz de confusión e informe de dirección sobre un ejemplo a mano.

import { Tensor } from "../../src/core/tensor.ts";
import { directionReport, errorsByDirection } from "../../src/sets/direction.ts";

const prediction = Tensor.from([1, -1, 1, -1, 0], [5]);
const target = Tensor.from([1, 1, -1, -1, 0], [5]);
const report = directionReport(prediction, target);
const errors = errorsByDirection(prediction, target);

console.log(
    `matriz: TP ${report.matrix.truePositive} · FP ${report.matrix.falsePositive} · ` +
        `TN ${report.matrix.trueNegative} · FN ${report.matrix.falseNegative} · ` +
        `empates ${report.matrix.ties}`,
);
console.log(
    `cobertura ${report.coverage.toFixed(2)} · accuracy ${report.accuracy.toFixed(2)} · ` +
        `precisión ${report.precision.toFixed(2)} · recall ${report.recall.toFixed(2)} · ` +
        `especificidad ${report.specificity.toFixed(2)}`,
);
console.log(
    `F1 ${report.f1.toFixed(2)} · balanced ${report.balancedAccuracy.toFixed(2)} · ` +
        `MCC ${report.mcc.toFixed(2)}`,
);
console.log(
    `tasa sube: predicha ${report.predictedUpRate.toFixed(2)} · ` +
        `real ${report.actualUpRate.toFixed(2)}`,
);
console.log(
    `MAE por dirección: sube ${errors.up.toFixed(2)} · baja ${errors.down.toFixed(2)}`,
);
