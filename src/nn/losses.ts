// Teoría: docs/teoria/entrenamiento/perdidas.md

import { Tensor } from "../core/tensor.ts";
import { mean, mul, sub } from "../core/tensor-ops.ts";

/** Error cuadrático medio; predicción y objetivo deben tener la misma forma. */
export function mse(prediction: Tensor, target: Tensor): Tensor {
    const sameShape =
        prediction.shape.length === target.shape.length &&
        prediction.shape.every((dimension, axis) => dimension === target.shape[axis]);
    if (!sameShape) {
        throw new Error(
            `mse espera formas iguales, no ${JSON.stringify(prediction.shape)} y ${JSON.stringify(target.shape)}`,
        );
    }
    const difference = sub(prediction, target);
    return mean(mul(difference, difference));
}

/** Entropía cruzada multiclase: logits [lote, clases] y la clase correcta por fila. */
export function crossEntropy(logits: Tensor, targets: readonly number[]): Tensor {
    if (logits.rank !== 2) {
        throw new Error("crossEntropy espera logits de rango 2 [lote, clases]");
    }
    const batch = logits.shape[0] as number;
    const classes = logits.shape[1] as number;
    if (classes <= 0) throw new Error("crossEntropy necesita al menos una clase");
    if (batch === 0) throw new Error("crossEntropy necesita un lote no vacío");
    if (targets.length !== batch) {
        throw new Error(`se esperaban ${batch} objetivos, no ${targets.length}`);
    }
    targets.forEach((target, row) => {
        if (!Number.isInteger(target) || target < 0 || target >= classes) {
            throw new RangeError(`objetivo ${target} fuera de [0, ${classes}) en la fila ${row}`);
        }
    });

    // Forward estable: restar el máximo por fila antes de exponenciar.
    const exponentials = new Float32Array(batch * classes);
    let total = 0;
    for (let row = 0; row < batch; row++) {
        let maximum = -Infinity;
        for (let column = 0; column < classes; column++) {
            maximum = Math.max(maximum, logits.get(row, column));
        }
        let sum = 0;
        for (let column = 0; column < classes; column++) {
            const value = Math.exp(logits.get(row, column) - maximum);
            exponentials[row * classes + column] = value;
            sum += value;
        }
        total += maximum + Math.log(sum) - logits.get(row, targets[row] as number);
    }

    const result = Tensor.from([total / batch], []);
    if (logits.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([logits], (gradient) => {
            const upstream = gradient.get();
            const values = new Float32Array(batch * classes);
            for (let row = 0; row < batch; row++) {
                let sum = 0;
                for (let column = 0; column < classes; column++) {
                    sum += exponentials[row * classes + column] as number;
                }
                for (let column = 0; column < classes; column++) {
                    const probability = (exponentials[row * classes + column] as number) / sum;
                    const oneHot = column === targets[row] ? 1 : 0;
                    values[row * classes + column] = ((probability - oneHot) * upstream) / batch;
                }
            }
            logits.addGrad(new Tensor(values, [batch, classes]));
        });
    }
    return result;
}
