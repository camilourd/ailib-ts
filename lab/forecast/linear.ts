// Teoría: docs/teoria/entrenamiento/pronostico.md
// Experimento: pronóstico lineal del retorno del día siguiente frente al baseline de predecir 0.

import { Tensor } from "../../src/core/tensor.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { directionReport, errorsByDirection } from "../../src/sets/direction.ts";
import {
    coefficientOfDetermination,
    correlation,
    meanAbsoluteError,
    meanError,
    rootMeanSquaredError,
} from "../../src/sets/metrics.ts";
import { syntheticAssetSet } from "../../src/sets/tabular/synthetic-asset.ts";
import { train } from "../../src/train/trainer.ts";

const set = syntheticAssetSet({ days: 1200, seed: 7 });
const linear = new Linear(3, 1, { seed: 1 });
const optimizer = new SGD(linear.parameters(), 0.05);
const history = train(
    (input) => linear.forward(input),
    set.train.input,
    {
        steps: 1000,
        optimizer,
        loss: (prediction) => set.loss(prediction, set.train),
    },
);

const prediction = linear.forward(set.test.input);
const baseline = Tensor.zeros(set.test.target.shape);

const build = (name: string, values: Tensor) => ({
    name,
    mae: meanAbsoluteError(values, set.test.target),
    rmse: rootMeanSquaredError(values, set.test.target),
    r2: coefficientOfDetermination(values, set.test.target),
    bias: meanError(values, set.test.target),
    correlation: correlation(values, set.test.target),
    direction: directionReport(values, set.test.target),
    errors: errorsByDirection(values, set.test.target),
});
const rows = [build("baseline", baseline), build("lineal", prediction)];

console.log(`set: ${set.name} · ${set.description}`);
console.log(
    `pasos ${history.length} · pérdida final ${(history.at(-1) as number).toFixed(4)} · ` +
        `cobertura ${rows[0]?.direction.coverage.toFixed(2)}`,
);
console.log("");
console.log("modelo     MAE      RMSE     R²       sesgo    corr");
for (const row of rows) {
    console.log(
        `${row.name.padEnd(9)} ${row.mae.toFixed(4)}   ${row.rmse.toFixed(4)}   ` +
            `${row.r2.toFixed(4)}   ${row.bias.toFixed(4)}   ${row.correlation.toFixed(4)}`,
    );
}
console.log("");
console.log("modelo     acc    prec   rec    espec  F1     bal    MCC    subePred subeReal");
for (const row of rows) {
    const report = row.direction;
    console.log(
        `${row.name.padEnd(9)} ${report.accuracy.toFixed(2)}   ${report.precision.toFixed(2)}   ` +
            `${report.recall.toFixed(2)}   ${report.specificity.toFixed(2)}   ` +
            `${report.f1.toFixed(2)}   ${report.balancedAccuracy.toFixed(2)}   ` +
            `${report.mcc.toFixed(2)}   ${report.predictedUpRate.toFixed(2)}     ` +
            `${report.actualUpRate.toFixed(2)}`,
    );
}
console.log("");
console.log("modelo     MAE sube  MAE baja  días sube  días baja");
for (const row of rows) {
    console.log(
        `${row.name.padEnd(9)} ${row.errors.up.toFixed(4)}    ${row.errors.down.toFixed(4)}    ` +
            `${row.errors.upCount}          ${row.errors.downCount}`,
    );
}
