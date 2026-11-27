// Teoría: docs/teoria/entrenamiento/sgd.md

import { Tensor } from "../core/tensor.ts";

/** Contrato de un optimizador: actualiza los parámetros y limpia sus gradientes. */
export interface Optimizer {
    step(): void;
    zeroGrad(): void;
}

export interface SGDOptions {
    weightDecay?: number;
}

/** Descenso de gradiente estocástico: θ ← θ − η · (∇L + λθ). */
export class SGD implements Optimizer {
    readonly #parameters: readonly Tensor[];
    readonly #weightDecay: number;
    #learningRate: number;

    constructor(parameters: readonly Tensor[], learningRate: number, options: SGDOptions = {}) {
        if (!Number.isFinite(learningRate) || learningRate <= 0) {
            throw new RangeError(`la tasa de aprendizaje debe ser un número positivo, no ${learningRate}`);
        }
        const weightDecay = options.weightDecay ?? 0;
        if (!Number.isFinite(weightDecay) || weightDecay < 0) {
            throw new RangeError(`weight decay debe ser ≥ 0, no ${weightDecay}`);
        }
        this.#parameters = parameters;
        this.#weightDecay = weightDecay;
        this.#learningRate = learningRate;
    }

    get learningRate(): number {
        return this.#learningRate;
    }

    set learningRate(value: number) {
        if (!Number.isFinite(value) || value <= 0) {
            throw new RangeError(`la tasa de aprendizaje debe ser un número positivo, no ${value}`);
        }
        this.#learningRate = value;
    }

    step(): void {
        for (const parameter of this.#parameters) {
            const gradient = parameter.grad;
            if (parameter.requiresGrad && gradient !== null) {
                for (let index = 0; index < parameter.size; index++) {
                    const gradientValue = gradient.data[index] as number;
                    const value = parameter.data[index] as number;
                    parameter.data[index] =
                        value -
                        this.#learningRate * (gradientValue + this.#weightDecay * value);
                }
            }
        }
    }

    zeroGrad(): void {
        for (const parameter of this.#parameters) parameter.zeroGrad();
    }
}
