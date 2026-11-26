// Teoría: docs/teoria/entrenamiento/pronostico.md
// Experimento: lineal y MLP con selección de hiperparámetros por validación; test solo al final.

import { Tensor } from "../../src/core/tensor.ts";
import { Linear } from "../../src/nn/linear.ts";
import { MLP } from "../../src/nn/mlp.ts";
import type { Model } from "../../src/nn/model.ts";
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
import type { Split } from "../../src/sets/types.ts";
import { train } from "../../src/train/trainer.ts";

const set = syntheticAssetSet({ days: 1200, seed: 7 });
const steps = 1000;

const trainModel = (model: Model, split: Split, learningRate: number): void => {
    const optimizer = new SGD(model.parameters(), learningRate);
    train((input) => model.forward(input), split.input, {
        steps,
        optimizer,
        loss: (prediction) => set.loss(prediction, split),
    });
};

const validationMae = (model: Model): number =>
    meanAbsoluteError(model.forward(set.validation.input), set.validation.target);

const concat = (first: Tensor, second: Tensor): Tensor => {
    const shape = [...first.shape];
    shape[0] = (first.shape[0] as number) + (second.shape[0] as number);
    const data = new Float32Array(first.size + second.size);
    data.set(first.data);
    data.set(second.data, first.size);
    return new Tensor(data, shape);
};

const finalSplit: Split = {
    input: concat(set.train.input, set.validation.input),
    target: concat(set.train.target, set.validation.target),
};

const linearRuns = [0.01, 0.05, 0.1, 0.5].map((learningRate) => {
    const model = new Linear(3, 1, { seed: 1 });
    trainModel(model, set.train, learningRate);
    return { learningRate, model, mae: validationMae(model) };
});
const bestLinear = linearRuns.reduce((best, current) => (current.mae < best.mae ? current : best));

const mlpRuns = [4, 8, 16].flatMap((hidden) =>
    [0.01, 0.05, 0.1].map((learningRate) => {
        const model = new MLP([3, hidden, 1], { seed: 2 });
        trainModel(model, set.train, learningRate);
        return { hidden, learningRate, model, mae: validationMae(model) };
    }),
);
const bestMlp = mlpRuns.reduce((best, current) => (current.mae < best.mae ? current : best));

const finalLinear = new Linear(3, 1, { seed: 1 });
trainModel(finalLinear, finalSplit, bestLinear.learningRate);
const finalMlp = new MLP([3, bestMlp.hidden, 1], { seed: 2 });
trainModel(finalMlp, finalSplit, bestMlp.learningRate);

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

const rows = [
    build("baseline", Tensor.zeros(set.test.target.shape)),
    build(`lineal lr=${bestLinear.learningRate}`, finalLinear.forward(set.test.input)),
    build(`mlp ${bestMlp.hidden} lr=${bestMlp.learningRate}`, finalMlp.forward(set.test.input)),
];

console.log(`set: ${set.name} · ${set.description}`);
console.log(
    `selección por validación: lineal lr=${bestLinear.learningRate} (MAE ${bestLinear.mae.toFixed(4)}) · ` +
        `mlp ${bestMlp.hidden} lr=${bestMlp.learningRate} (MAE ${bestMlp.mae.toFixed(4)})`,
);
console.log(`cobertura de dirección: ${rows[0]?.direction.coverage.toFixed(2)}`);
console.log("");
console.log("modelo                 MAE      RMSE     R²       sesgo    corr");
for (const row of rows) {
    console.log(
        `${row.name.padEnd(22)} ${row.mae.toFixed(4)}   ${row.rmse.toFixed(4)}   ` +
            `${row.r2.toFixed(4)}   ${row.bias.toFixed(4)}   ${row.correlation.toFixed(4)}`,
    );
}
console.log("");
console.log("modelo                 acc    prec   rec    espec  F1     bal    MCC    subePred subeReal");
for (const row of rows) {
    const report = row.direction;
    console.log(
        `${row.name.padEnd(22)} ${report.accuracy.toFixed(2)}   ${report.precision.toFixed(2)}   ` +
            `${report.recall.toFixed(2)}   ${report.specificity.toFixed(2)}   ` +
            `${report.f1.toFixed(2)}   ${report.balancedAccuracy.toFixed(2)}   ` +
            `${report.mcc.toFixed(2)}   ${report.predictedUpRate.toFixed(2)}     ` +
            `${report.actualUpRate.toFixed(2)}`,
    );
}
console.log("");
console.log("modelo                 MAE sube  MAE baja  días sube  días baja");
for (const row of rows) {
    console.log(
        `${row.name.padEnd(22)} ${row.errors.up.toFixed(4)}    ${row.errors.down.toFixed(4)}    ` +
            `${row.errors.upCount}          ${row.errors.downCount}`,
    );
}
