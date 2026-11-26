// Teoría: docs/teoria/entrenamiento/pronostico-secuencial.md
// Experimento: RNN, GRU, LSTM y transformer sobre el activo sintético del primer set.

import { createRng } from "../../src/core/random/index.ts";
import { Tensor } from "../../src/core/tensor.ts";
import { mse } from "../../src/nn/losses.ts";
import type { Model } from "../../src/nn/model.ts";
import { GRU } from "../../src/nn/gru.ts";
import { Linear } from "../../src/nn/linear.ts";
import { LSTM } from "../../src/nn/lstm.ts";
import type { RecurrentCell } from "../../src/nn/recurrent.ts";
import { RNN } from "../../src/nn/rnn.ts";
import { TransformerBlock } from "../../src/nn/transformer-block.ts";
import { sliceRows } from "../../src/core/tensor-ops.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { directionReport } from "../../src/sets/direction.ts";
import {
    coefficientOfDetermination,
    correlation,
    meanAbsoluteError,
    rootMeanSquaredError,
} from "../../src/sets/metrics.ts";
import { writeFileSync } from "node:fs";

const DAYS = 1200;
const WINDOW = 10;
const FEATURES = 3;
const TRAIN_FRACTION = 0.7;
const VALIDATION_FRACTION = 0.85;
const STEPS = 1000;
const LEARNING_RATE = 0.005;
const SEEDS = [1, 2, 3];

interface Split {
    input: Tensor;
    target: Tensor;
}

function buildSet(seed: number): { train: Split; validation: Split; test: Split } {
    const rng = createRng(seed);
    const normal = (): number => {
        const first = 1 - rng.next();
        const second = rng.next();
        return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
    };
    const returns1 = new Float32Array(DAYS);
    const returns2 = new Float32Array(DAYS);
    for (let day = 0; day < DAYS; day++) {
        const common = normal();
        returns1[day] = 0.0003 + 0.01 * (0.6 * common + 0.8 * normal());
        returns2[day] = 0.0002 + 0.012 * (0.5 * common + 0.866 * normal());
    }

    const rows = DAYS - WINDOW - 1;
    const raw = new Float32Array(rows * WINDOW * FEATURES);
    const targets = new Float32Array(rows);
    for (let row = 0; row < rows; row++) {
        const day = row + WINDOW - 1;
        for (let offset = 0; offset < WINDOW; offset++) {
            const current = day - WINDOW + 1 + offset;
            let sum = 0;
            let squares = 0;
            let count = 0;
            for (let index = Math.max(0, current - WINDOW + 1); index <= current; index++) {
                const value = returns1[index] as number;
                sum += value;
                squares += value * value;
                count += 1;
            }
            const mean = sum / count;
            const variance = Math.max(squares / count - mean * mean, 0);
            const base = (row * WINDOW + offset) * FEATURES;
            raw[base] = returns1[current] as number;
            raw[base + 1] = Math.sqrt(variance);
            raw[base + 2] = returns2[current] as number;
        }
        targets[row] = (returns1[day + 1] as number) * 100;
    }

    const trainEnd = Math.floor(rows * TRAIN_FRACTION);
    const validationEnd = Math.floor(rows * VALIDATION_FRACTION);
    const means = [0, 0, 0];
    const deviations = [0, 0, 0];
    for (let row = 0; row < trainEnd; row++) {
        for (let offset = 0; offset < WINDOW; offset++) {
            for (let feature = 0; feature < FEATURES; feature++) {
                means[feature] =
                    (means[feature] as number) +
                    (raw[(row * WINDOW + offset) * FEATURES + feature] as number);
            }
        }
    }
    for (let feature = 0; feature < FEATURES; feature++) {
        means[feature] = (means[feature] as number) / (trainEnd * WINDOW);
    }
    for (let row = 0; row < trainEnd; row++) {
        for (let offset = 0; offset < WINDOW; offset++) {
            for (let feature = 0; feature < FEATURES; feature++) {
                const value = raw[(row * WINDOW + offset) * FEATURES + feature] as number;
                const delta = value - (means[feature] as number);
                deviations[feature] = (deviations[feature] as number) + delta * delta;
            }
        }
    }
    for (let feature = 0; feature < FEATURES; feature++) {
        const deviation = Math.sqrt((deviations[feature] as number) / (trainEnd * WINDOW));
        deviations[feature] = deviation > 0 ? deviation : 1;
    }
    for (let index = 0; index < raw.length; index++) {
        const feature = index % FEATURES;
        raw[index] =
            ((raw[index] as number) - (means[feature] as number)) / (deviations[feature] as number);
    }

    const make = (start: number, end: number): Split => {
        const count = end - start;
        const input = new Float32Array(count * WINDOW * FEATURES);
        for (let row = 0; row < count; row++) {
            input.set(raw.subarray((start + row) * WINDOW * FEATURES, (start + row + 1) * WINDOW * FEATURES), row * WINDOW * FEATURES);
        }
        return {
            input: new Tensor(input, [count, WINDOW, FEATURES]),
            target: new Tensor(targets.slice(start, end), [count, 1]),
        };
    };
    return {
        train: make(0, trainEnd),
        validation: make(trainEnd, validationEnd),
        test: make(validationEnd, rows),
    };
}

class RecurrentForecaster implements Model {
    readonly #cell: RecurrentCell;
    readonly #output: Linear;

    constructor(cell: RecurrentCell, seed: number) {
        this.#cell = cell;
        this.#output = new Linear(cell.hiddenSize, 1, { seed });
    }

    forward(input: Tensor): Tensor {
        let state = this.#cell.initial();
        const features = input.shape[1] as number;
        for (let position = 0; position < (input.shape[0] as number); position++) {
            const row = new Tensor(
                input.data.subarray(position * features, (position + 1) * features),
                [1, features],
            );
            state = this.#cell.advance(row, state);
        }
        return this.#output.forward(state.hidden);
    }

    parameters(): readonly Tensor[] {
        return [...this.#cell.parameters(), ...this.#output.parameters()];
    }
}

class TransformerForecaster implements Model {
    readonly #input: Linear;
    readonly #blocks: TransformerBlock[];
    readonly #output: Linear;

    constructor(seed: number) {
        this.#input = new Linear(FEATURES, 32, { seed });
        this.#blocks = [0, 1].map(
            (index) => new TransformerBlock(32, 4, { hiddenSize: 64, seed: seed + index * 10, rope: true }),
        );
        this.#output = new Linear(32, 1, { seed: seed + 20 });
    }

    forward(input: Tensor): Tensor {
        let value = this.#input.forward(input);
        for (const block of this.#blocks) value = block.forward(value).output;
        const last = sliceRows(value, (value.shape[0] as number) - 1, value.shape[0] as number);
        return this.#output.forward(last);
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.#input.parameters(),
            ...this.#blocks.flatMap((block) => block.parameters()),
            ...this.#output.parameters(),
        ];
    }
}

function trainModel(model: Model, split: Split, seed: number): void {
    const optimizer = new SGD(model.parameters(), LEARNING_RATE);
    const rng = createRng(seed + 100);
    const rows = split.target.shape[0] as number;
    for (let step = 0; step < STEPS; step++) {
        const index = rng.int(rows);
        const input = new Tensor(
            split.input.data.subarray(
                index * WINDOW * FEATURES,
                (index + 1) * WINDOW * FEATURES,
            ),
            [WINDOW, FEATURES],
        );
        const target = new Tensor(split.target.data.subarray(index, index + 1), [1, 1]);
        optimizer.zeroGrad();
        const loss = mse(model.forward(input), target);
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }
}

function predict(model: Model, split: Split): Float32Array {
    const rows = split.target.shape[0] as number;
    const predictions = new Float32Array(rows);
    for (let index = 0; index < rows; index++) {
        const input = new Tensor(
            split.input.data.subarray(index * WINDOW * FEATURES, (index + 1) * WINDOW * FEATURES),
            [WINDOW, FEATURES],
        );
        predictions[index] = model.forward(input).data[0] as number;
    }
    return predictions;
}

function metrics(predictions: Float32Array, split: Split): {
    mae: number;
    rmse: number;
    r2: number;
    correlation: number;
    direction: ReturnType<typeof directionReport>;
} {
    const predicted = new Tensor(predictions, [predictions.length, 1]);
    return {
        mae: meanAbsoluteError(predicted, split.target),
        rmse: rootMeanSquaredError(predicted, split.target),
        r2: coefficientOfDetermination(predicted, split.target),
        correlation: correlation(predicted, split.target),
        direction: directionReport(predicted, split.target),
    };
}

const set = buildSet(7);
const baselinePredictions = new Float32Array(set.test.target.shape[0] as number);
const rows: Array<Record<string, unknown>> = [];
const baseline = metrics(baselinePredictions, set.test);
console.log(
    `activo sintético · ventana ${WINDOW} · ${set.train.target.shape[0]} train / ` +
        `${set.validation.target.shape[0]} val / ${set.test.target.shape[0]} test · ` +
        `baseline MAE ${baseline.mae.toFixed(4)}`,
);
console.log("modelo        params    MAE (media ± sd)      RMSE     R²       acc    MCC    TP  FP  TN  FN");
const baselineRow = {
    name: "cero",
    parameters: 0,
    mae: baseline.mae,
    maeStd: 0,
    rmse: baseline.rmse,
    r2: baseline.r2,
    accuracy: baseline.direction.accuracy,
    mcc: baseline.direction.mcc,
    matrix: baseline.direction.matrix,
};
rows.push(baselineRow);
console.log(
    `cero          ${"0".padStart(7)}   ${baseline.mae.toFixed(4)} ± 0.0000      ` +
        `${baseline.rmse.toFixed(4)}   ${baseline.r2.toFixed(4)}   ` +
        `${baseline.direction.accuracy.toFixed(2)}   ${baseline.direction.mcc.toFixed(2)}   ` +
        `${baseline.direction.matrix.truePositive}   ${baseline.direction.matrix.falsePositive}   ` +
        `${baseline.direction.matrix.trueNegative}   ${baseline.direction.matrix.falseNegative}`,
);

const factories: Array<{ name: string; make: (seed: number) => Model }> = [
    { name: "rnn", make: (seed) => new RecurrentForecaster(new RNN(FEATURES, 16, { seed }), seed + 1) },
    { name: "gru", make: (seed) => new RecurrentForecaster(new GRU(FEATURES, 16, { seed }), seed + 1) },
    { name: "lstm", make: (seed) => new RecurrentForecaster(new LSTM(FEATURES, 16, { seed }), seed + 1) },
    { name: "transformer", make: (seed) => new TransformerForecaster(seed) },
];

for (const factory of factories) {
    const seedMetrics: ReturnType<typeof metrics>[] = [];
    const allPredictions: number[] = [];
    const allTargets: number[] = [];
    let parameters = 0;
    for (const seed of SEEDS) {
        const model = factory.make(seed);
        parameters = model.parameters().reduce((total, parameter) => total + parameter.size, 0);
        trainModel(model, set.train, seed);
        const predictions = predict(model, set.test);
        seedMetrics.push(metrics(predictions, set.test));
        allPredictions.push(...predictions);
        allTargets.push(...set.test.target.data);
    }
    const maes = seedMetrics.map((item) => item.mae);
    const meanMae = maes.reduce((total, value) => total + value, 0) / maes.length;
    const variance = maes.reduce((total, value) => total + (value - meanMae) ** 2, 0) / maes.length;
    const direction = directionReport(
        new Tensor(Float32Array.from(allPredictions), [allPredictions.length, 1]),
        new Tensor(Float32Array.from(allTargets), [allTargets.length, 1]),
    );
    const mean = (values: number[]): number => values.reduce((total, value) => total + value, 0) / values.length;
    const row = {
        name: factory.name,
        parameters,
        mae: meanMae,
        maeStd: Math.sqrt(variance),
        rmse: mean(seedMetrics.map((item) => item.rmse)),
        r2: mean(seedMetrics.map((item) => item.r2)),
        accuracy: direction.accuracy,
        mcc: direction.mcc,
        matrix: direction.matrix,
    };
    rows.push(row);
    console.log(
        `${factory.name.padEnd(13)} ${String(parameters).padStart(6)}   ` +
            `${row.mae.toFixed(4)} ± ${row.maeStd.toFixed(4)}      ` +
            `${row.rmse.toFixed(4)}   ${row.r2.toFixed(4)}   ` +
            `${row.accuracy.toFixed(2)}   ${row.mcc.toFixed(2)}   ` +
            `${direction.matrix.truePositive}   ${direction.matrix.falsePositive}   ` +
            `${direction.matrix.trueNegative}   ${direction.matrix.falseNegative}`,
    );
}

writeFileSync(
    "data/forecast-sequence.json",
    `${JSON.stringify(
        {
            set: "activo-sintetico-secuencial",
            window: WINDOW,
            features: FEATURES,
            seeds: SEEDS,
            steps: STEPS,
            learningRate: LEARNING_RATE,
            rows,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/forecast-sequence.json");
