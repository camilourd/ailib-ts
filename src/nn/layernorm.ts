// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";

export interface LayerNormOptions {
    epsilon?: number;
}

/** LayerNorm por fila: normaliza la última dimensión y aplica gamma/beta. */
export class LayerNorm {
    readonly gamma: Tensor;
    readonly beta: Tensor;
    readonly #size: number;
    readonly #epsilon: number;

    constructor(size: number, options: LayerNormOptions = {}) {
        if (!Number.isInteger(size) || size <= 0) {
            throw new Error(`LayerNorm necesita una dimensión positiva, no ${size}`);
        }
        const epsilon = options.epsilon ?? 1e-5;
        if (!Number.isFinite(epsilon) || epsilon <= 0) {
            throw new RangeError(`epsilon debe ser positivo, no ${epsilon}`);
        }
        this.gamma = Tensor.ones([size], { requiresGrad: true });
        this.beta = Tensor.zeros([size], { requiresGrad: true });
        this.#size = size;
        this.#epsilon = epsilon;
    }

    forward(input: Tensor): Tensor {
        if (input.rank !== 2 || (input.shape[1] as number) !== this.#size) {
            throw new Error(
                `LayerNorm espera [filas, ${this.#size}], no ${JSON.stringify(input.shape)}`,
            );
        }
        const rows = input.shape[0] as number;
        const size = this.#size;
        const normalized = new Float32Array(rows * size);
        const inverseDeviation = new Float32Array(rows);
        const output = new Float32Array(rows * size);
        for (let row = 0; row < rows; row++) {
            let sum = 0;
            for (let column = 0; column < size; column++) {
                sum += input.data[row * size + column] as number;
            }
            const mean = sum / size;
            let squares = 0;
            for (let column = 0; column < size; column++) {
                const delta = (input.data[row * size + column] as number) - mean;
                squares += delta * delta;
            }
            const variance = squares / size;
            const inverse = 1 / Math.sqrt(variance + this.#epsilon);
            inverseDeviation[row] = inverse;
            for (let column = 0; column < size; column++) {
                const value =
                    ((input.data[row * size + column] as number) - mean) * inverse;
                normalized[row * size + column] = value;
                output[row * size + column] =
                    value * (this.gamma.data[column] as number) +
                    (this.beta.data[column] as number);
            }
        }

        const result = new Tensor(output, [rows, size]);
        if (input.requiresGrad || this.gamma.requiresGrad || this.beta.requiresGrad) {
            result.requiresGrad = true;
            result.setGraph([input, this.gamma, this.beta], (gradient) => {
                const dGamma = new Float32Array(size);
                const dBeta = new Float32Array(size);
                const dNormalized = new Float32Array(rows * size);
                for (let row = 0; row < rows; row++) {
                    for (let column = 0; column < size; column++) {
                        const value = gradient.data[row * size + column] as number;
                        dGamma[column] = (dGamma[column] as number) +
                            value * (normalized[row * size + column] as number);
                        dBeta[column] = (dBeta[column] as number) + value;
                        dNormalized[row * size + column] =
                            value * (this.gamma.data[column] as number);
                    }
                }
                const dInput = new Float32Array(rows * size);
                for (let row = 0; row < rows; row++) {
                    let sumNormalized = 0;
                    let sumNormalizedSquared = 0;
                    for (let column = 0; column < size; column++) {
                        const value = dNormalized[row * size + column] as number;
                        sumNormalized += value;
                        sumNormalizedSquared +=
                            value * (normalized[row * size + column] as number);
                    }
                    const inverse = inverseDeviation[row] as number;
                    for (let column = 0; column < size; column++) {
                        const value = dNormalized[row * size + column] as number;
                        dInput[row * size + column] =
                            (inverse / size) *
                            (size * value - sumNormalized -
                                (normalized[row * size + column] as number) *
                                    sumNormalizedSquared);
                    }
                }
                input.addGrad(new Tensor(dInput, input.shape));
                this.gamma.addGrad(new Tensor(dGamma, [size]));
                this.beta.addGrad(new Tensor(dBeta, [size]));
            });
        }
        return result;
    }

    parameters(): readonly Tensor[] {
        return [this.gamma, this.beta];
    }
}
