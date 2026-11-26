// Teoría: docs/teoria/entrenamiento/direccion.md

import type { Tensor } from "../core/tensor.ts";

export interface ConfusionMatrix {
    truePositive: number;
    falsePositive: number;
    trueNegative: number;
    falseNegative: number;
    ties: number;
    total: number;
}

export interface DirectionReport {
    matrix: ConfusionMatrix;
    evaluated: number;
    coverage: number;
    accuracy: number;
    precision: number;
    recall: number;
    specificity: number;
    f1: number;
    balancedAccuracy: number;
    mcc: number;
    predictedUpRate: number;
    actualUpRate: number;
}

export interface DirectionalErrors {
    up: number;
    down: number;
    upCount: number;
    downCount: number;
}

function checkedSize(prediction: Tensor, target: Tensor): number {
    const sameShape =
        prediction.shape.length === target.shape.length &&
        prediction.shape.every((dimension, axis) => dimension === target.shape[axis]);
    if (!sameShape) {
        throw new Error(
            `formas distintas: ${JSON.stringify(prediction.shape)} y ${JSON.stringify(target.shape)}`,
        );
    }
    if (prediction.size === 0) throw new Error("no se puede medir una predicción vacía");
    return prediction.size;
}

/**
 * Matriz de confusión de la dirección. La clase positiva es «sube» (retorno > 0); los objetivos
 * exactamente 0 se cuentan como `ties` y quedan fuera de la matriz. Una predicción 0 cuenta como
 * «baja».
 */
export function confusionMatrix(prediction: Tensor, target: Tensor): ConfusionMatrix {
    const size = checkedSize(prediction, target);
    let truePositive = 0;
    let falsePositive = 0;
    let trueNegative = 0;
    let falseNegative = 0;
    let ties = 0;
    for (let index = 0; index < size; index++) {
        const actual = target.data[index] as number;
        const predicted = prediction.data[index] as number;
        if (actual === 0) {
            ties += 1;
        } else {
            const actualUp = actual > 0;
            const predictedUp = predicted > 0;
            if (actualUp && predictedUp) truePositive += 1;
            else if (actualUp) falseNegative += 1;
            else if (predictedUp) falsePositive += 1;
            else trueNegative += 1;
        }
    }
    return { truePositive, falsePositive, trueNegative, falseNegative, ties, total: size };
}

export function directionReport(prediction: Tensor, target: Tensor): DirectionReport {
    const matrix = confusionMatrix(prediction, target);
    const evaluated = matrix.total - matrix.ties;
    const divide = (numerator: number, denominator: number): number =>
        denominator > 0 ? numerator / denominator : 0;

    const accuracy = divide(matrix.truePositive + matrix.trueNegative, evaluated);
    const precision = divide(matrix.truePositive, matrix.truePositive + matrix.falsePositive);
    const recall = divide(matrix.truePositive, matrix.truePositive + matrix.falseNegative);
    const specificity = divide(matrix.trueNegative, matrix.trueNegative + matrix.falsePositive);
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    const balancedAccuracy = (recall + specificity) / 2;
    const mccDenominator = Math.sqrt(
        (matrix.truePositive + matrix.falsePositive) *
            (matrix.truePositive + matrix.falseNegative) *
            (matrix.trueNegative + matrix.falsePositive) *
            (matrix.trueNegative + matrix.falseNegative),
    );
    const mcc =
        mccDenominator > 0
            ? (matrix.truePositive * matrix.trueNegative -
                  matrix.falsePositive * matrix.falseNegative) /
              mccDenominator
            : 0;

    return {
        matrix,
        evaluated,
        coverage: divide(evaluated, matrix.total),
        accuracy,
        precision,
        recall,
        specificity,
        f1,
        balancedAccuracy,
        mcc,
        predictedUpRate: divide(matrix.truePositive + matrix.falsePositive, evaluated),
        actualUpRate: divide(matrix.truePositive + matrix.falseNegative, evaluated),
    };
}

/** MAE separado por la dirección real; los ceros quedan fuera. */
export function errorsByDirection(prediction: Tensor, target: Tensor): DirectionalErrors {
    const size = checkedSize(prediction, target);
    let upError = 0;
    let downError = 0;
    let upCount = 0;
    let downCount = 0;
    for (let index = 0; index < size; index++) {
        const actual = target.data[index] as number;
        if (actual !== 0) {
            const error = Math.abs((prediction.data[index] as number) - actual);
            if (actual > 0) {
                upError += error;
                upCount += 1;
            } else {
                downError += error;
                downCount += 1;
            }
        }
    }
    return {
        up: upCount > 0 ? upError / upCount : 0,
        down: downCount > 0 ? downError / downCount : 0,
        upCount,
        downCount,
    };
}
