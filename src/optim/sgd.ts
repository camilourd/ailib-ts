// Teoría: docs/teoria/entrenamiento/sgd.md

import { Tensor } from "../core/tensor.ts";

/** Contrato de un optimizador: actualiza los parámetros y limpia sus gradientes. */
export interface Optimizer {
    step(): void;
    zeroGrad(): void;
}

/** Descenso de gradiente estocástico: θ ← θ − η · ∇L. */
export class SGD implements Optimizer {
    readonly #parameters: readonly Tensor[];
    readonly #learningRate: number;

    constructor(parameters: readonly Tensor[], learningRate: number) {
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new RangeError(`la tasa de aprendizaje debe ser un número positivo, no ${learningRate}`);
        }
        this.#parameters = parameters;
        this.#learningRate = learningRate;
    }

    get learningRate(): number {
        return this.#learningRate;
    }

    step(): void {
        for (const parameter of this.#parameters) {
            const gradient = parameter.grad;
            if (parameter.requiresGrad && gradient !== null) {
                for (let index = 0; index < parameter.size; index++) {
                    parameter.data[index] =
                        (parameter.data[index] as number) -
                        this.#learningRate * (gradient.data[index] as number);
                }
            }
        }
    }

    zeroGrad(): void {
        for (const parameter of this.#parameters) parameter.zeroGrad();
    }
}
