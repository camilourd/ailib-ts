// Teoría: docs/teoria/entrenamiento/sets.md

import type { Tensor } from "../core/tensor.ts";

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

export function meanAbsoluteError(prediction: Tensor, target: Tensor): number {
    const size = checkedSize(prediction, target);
    let total = 0;
    for (let index = 0; index < size; index++) {
        total += Math.abs((prediction.data[index] as number) - (target.data[index] as number));
    }
    return total / size;
}

export function rootMeanSquaredError(prediction: Tensor, target: Tensor): number {
    const size = checkedSize(prediction, target);
    let total = 0;
    for (let index = 0; index < size; index++) {
        const difference = (prediction.data[index] as number) - (target.data[index] as number);
        total += difference * difference;
    }
    return Math.sqrt(total / size);
}

/** Proporción de signos que coinciden; un cero solo acierta contra otro cero. */
export function directionalAccuracy(prediction: Tensor, target: Tensor): number {
    const size = checkedSize(prediction, target);
    let correct = 0;
    for (let index = 0; index < size; index++) {
        if (
            Math.sign(prediction.data[index] as number) === Math.sign(target.data[index] as number)
        ) {
            correct += 1;
        }
    }
    return correct / size;
}

/** Sesgo: media de (predicción − objetivo); positivo = optimista. */
export function meanError(prediction: Tensor, target: Tensor): number {
    const size = checkedSize(prediction, target);
    let total = 0;
    for (let index = 0; index < size; index++) {
        total += (prediction.data[index] as number) - (target.data[index] as number);
    }
    return total / size;
}

/** Coeficiente de determinación R²; 0 si el objetivo no tiene varianza. */
export function coefficientOfDetermination(prediction: Tensor, target: Tensor): number {
    const size = checkedSize(prediction, target);
    let mean = 0;
    for (let index = 0; index < size; index++) mean += target.data[index] as number;
    mean /= size;
    let residual = 0;
    let total = 0;
    for (let index = 0; index < size; index++) {
        const difference = (prediction.data[index] as number) - (target.data[index] as number);
        residual += difference * difference;
        const centered = (target.data[index] as number) - mean;
        total += centered * centered;
    }
    return total > 0 ? 1 - residual / total : 0;
}

/** Correlación de Pearson entre predicción y objetivo; 0 si alguna no tiene varianza. */
export function correlation(prediction: Tensor, target: Tensor): number {
    const size = checkedSize(prediction, target);
    let meanPrediction = 0;
    let meanTarget = 0;
    for (let index = 0; index < size; index++) {
        meanPrediction += prediction.data[index] as number;
        meanTarget += target.data[index] as number;
    }
    meanPrediction /= size;
    meanTarget /= size;
    let covariance = 0;
    let variancePrediction = 0;
    let varianceTarget = 0;
    for (let index = 0; index < size; index++) {
        const predictionDelta = (prediction.data[index] as number) - meanPrediction;
        const targetDelta = (target.data[index] as number) - meanTarget;
        covariance += predictionDelta * targetDelta;
        variancePrediction += predictionDelta * predictionDelta;
        varianceTarget += targetDelta * targetDelta;
    }
    const denominator = Math.sqrt(variancePrediction * varianceTarget);
    return denominator > 0 ? covariance / denominator : 0;
}
