// Teoría: docs/teoria/entrenamiento/bucle.md

import type { Tensor } from "../core/tensor.ts";
import type { Optimizer } from "../optim/sgd.ts";

export interface TrainOptions {
    steps: number;
    optimizer: Optimizer;
    /** Pérdida para la predicción del paso; debe devolver un escalar. */
    loss: (prediction: Tensor) => Tensor;
    onStep?: (step: number, loss: number) => void;
}

/**
 * Bucle de entrenamiento: zeroGrad → forward → pérdida → backward → step.
 * Devuelve la pérdida de cada paso, en orden.
 */
export function train(
    forward: (input: Tensor) => Tensor,
    input: Tensor,
    options: TrainOptions,
): number[] {
    if (!Number.isInteger(options.steps) || options.steps < 0) {
        throw new Error(`el número de pasos debe ser un entero ≥ 0, no ${options.steps}`);
    }
    const history: number[] = [];
    for (let step = 0; step < options.steps; step++) {
        options.optimizer.zeroGrad();
        const loss = options.loss(forward(input));
        if (loss.rank !== 0) throw new Error("la pérdida debe ser un escalar");
        loss.backward();
        options.optimizer.step();
        const value = loss.get();
        history.push(value);
        options.onStep?.(step, value);
    }
    return history;
}
