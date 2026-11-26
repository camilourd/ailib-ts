// Teoría: docs/teoria/entrenamiento/sets.md

import { createRng } from "../../core/random/index.ts";
import { Tensor } from "../../core/tensor.ts";
import { mse } from "../../nn/losses.ts";
import { meanAbsoluteError } from "../metrics.ts";
import type { RegressionSet, Split } from "../types.ts";

const FEATURES = 3;
const WINDOW = 10;
const TRAIN_FRACTION = 0.7;
const VALIDATION_FRACTION = 0.85;

export interface SyntheticAssetOptions {
    days?: number;
    seed?: number;
}

/**
 * Set de regresión sintético: un activo con GBM, un segundo activo correlacionado y una volatilidad
 * móvil. El objetivo es el retorno del día siguiente en porcentaje.
 */
export function syntheticAssetSet(options: SyntheticAssetOptions = {}): RegressionSet {
    const days = options.days ?? 1200;
    const seed = options.seed ?? 1;
    if (!Number.isInteger(days) || days < WINDOW + 10) {
        throw new Error(`hacen falta al menos ${WINDOW + 10} días, no ${days}`);
    }

    const rng = createRng(seed);
    const normal = (): number => {
        const first = 1 - rng.next();
        const second = rng.next();
        return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
    };

    const returns1 = new Float32Array(days);
    const returns2 = new Float32Array(days);
    for (let day = 0; day < days; day++) {
        const common = normal();
        returns1[day] = 0.0003 + 0.01 * (0.6 * common + 0.8 * normal());
        returns2[day] = 0.0002 + 0.012 * (0.5 * common + 0.866 * normal());
    }

    const rows = days - WINDOW - 1;
    const features = new Float32Array(rows * FEATURES);
    const targets = new Float32Array(rows);
    for (let row = 0; row < rows; row++) {
        const day = row + WINDOW;
        let sum = 0;
        let squares = 0;
        for (let index = day - WINDOW + 1; index <= day; index++) {
            const value = returns1[index] as number;
            sum += value;
            squares += value * value;
        }
        const mean = sum / WINDOW;
        const variance = Math.max(squares / WINDOW - mean * mean, 0);
        features[row * FEATURES] = returns1[day] as number;
        features[row * FEATURES + 1] = Math.sqrt(variance);
        features[row * FEATURES + 2] = returns2[day] as number;
        targets[row] = (returns1[day + 1] as number) * 100;
    }

    const trainEnd = Math.floor(rows * TRAIN_FRACTION);
    const validationEnd = Math.floor(rows * VALIDATION_FRACTION);

    const means = [0, 0, 0];
    for (let row = 0; row < trainEnd; row++) {
        for (let feature = 0; feature < FEATURES; feature++) {
            means[feature] =
                (means[feature] as number) + (features[row * FEATURES + feature] as number);
        }
    }
    for (let feature = 0; feature < FEATURES; feature++) {
        means[feature] = (means[feature] as number) / trainEnd;
    }

    const deviations = [0, 0, 0];
    for (let row = 0; row < trainEnd; row++) {
        for (let feature = 0; feature < FEATURES; feature++) {
            const delta =
                (features[row * FEATURES + feature] as number) - (means[feature] as number);
            deviations[feature] = (deviations[feature] as number) + delta * delta;
        }
    }
    for (let feature = 0; feature < FEATURES; feature++) {
        const deviation = Math.sqrt((deviations[feature] as number) / trainEnd);
        deviations[feature] = deviation > 0 ? deviation : 1;
    }

    const normalized = new Float32Array(rows * FEATURES);
    for (let row = 0; row < rows; row++) {
        for (let feature = 0; feature < FEATURES; feature++) {
            normalized[row * FEATURES + feature] =
                ((features[row * FEATURES + feature] as number) - (means[feature] as number)) /
                (deviations[feature] as number);
        }
    }

    const split = (start: number, end: number): Split => ({
        input: new Tensor(normalized.slice(start * FEATURES, end * FEATURES), [
            end - start,
            FEATURES,
        ]),
        target: new Tensor(targets.slice(start, end), [end - start, 1]),
    });
    const train = split(0, trainEnd);
    const validation = split(trainEnd, validationEnd);
    const test = split(validationEnd, rows);

    let baseline = 0;
    for (const value of test.target.data) baseline += Math.abs(value);
    baseline /= test.target.size;

    return {
        name: "activo-sintetico",
        description: `GBM de ${days} días (semilla ${seed}); retorno del día siguiente en %`,
        kind: "regression",
        metricName: "MAE",
        train,
        validation,
        test,
        loss: (prediction, part) => mse(prediction, part.target),
        metric: (prediction, part) => meanAbsoluteError(prediction, part.target),
        baseline,
    };
}
