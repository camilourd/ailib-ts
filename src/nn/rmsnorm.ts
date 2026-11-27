// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";

export interface RMSNormOptions {
    epsilon?: number;
}

/** RMSNorm por fila: x / sqrt(mean(x²) + ε) · gamma. Sin media ni beta. */
export class RMSNorm {
    readonly gamma: Tensor;
    readonly #size: number;
    readonly #epsilon: number;

    constructor(size: number, options: RMSNormOptions = {}) {
        if (!Number.isInteger(size) || size <= 0) {
            throw new Error(`RMSNorm necesita una dimensión positiva, no ${size}`);
        }
        const epsilon = options.epsilon ?? 1e-6;
        if (!Number.isFinite(epsilon) || epsilon <= 0) {
            throw new RangeError(`epsilon debe ser positivo, no ${epsilon}`);
        }
        this.gamma = Tensor.ones([size], { requiresGrad: true });
        this.#size = size;
        this.#epsilon = epsilon;
    }

    forward(input: Tensor): Tensor {
        if (input.rank !== 2 || (input.shape[1] as number) !== this.#size) {
            throw new Error(
                `RMSNorm espera [filas, ${this.#size}], no ${JSON.stringify(input.shape)}`,
            );
        }
        const rows = input.shape[0] as number;
        const size = this.#size;
        const normalized = new Float32Array(rows * size);
        const inverse = new Float32Array(rows);
        const output = new Float32Array(rows * size);
        for (let row = 0; row < rows; row++) {
            let squares = 0;
            for (let column = 0; column < size; column++) {
                const value = input.data[row * size + column] as number;
                squares += value * value;
            }
            const rms = Math.sqrt(squares / size + this.#epsilon);
            const scale = 1 / rms;
            inverse[row] = scale;
            for (let column = 0; column < size; column++) {
                const value = (input.data[row * size + column] as number) * scale;
                normalized[row * size + column] = value;
                output[row * size + column] = value * (this.gamma.data[column] as number);
            }
        }

        const result = new Tensor(output, [rows, size]);
        if (input.requiresGrad || this.gamma.requiresGrad) {
            result.requiresGrad = true;
            result.setGraph([input, this.gamma], (gradient) => {
                const dGamma = new Float32Array(size);
                const dNormalized = new Float32Array(rows * size);
                const dInput = new Float32Array(rows * size);
                for (let row = 0; row < rows; row++) {
                    let dot = 0;
                    for (let column = 0; column < size; column++) {
                        const value = gradient.data[row * size + column] as number;
                        dGamma[column] = (dGamma[column] as number) +
                            value * (normalized[row * size + column] as number);
                        const normalizedGradient = value * (this.gamma.data[column] as number);
                        dNormalized[row * size + column] = normalizedGradient;
                        dot += normalizedGradient * (input.data[row * size + column] as number);
                    }
                    const scale = inverse[row] as number;
                    for (let column = 0; column < size; column++) {
                        dInput[row * size + column] =
                            scale * (dNormalized[row * size + column] as number) -
                            (input.data[row * size + column] as number) * dot * scale ** 3 / size;
                    }
                }
                input.addGrad(new Tensor(dInput, input.shape));
                this.gamma.addGrad(new Tensor(dGamma, [size]));
            });
        }
        return result;
    }

    parameters(): readonly Tensor[] {
        return [this.gamma];
    }
}
