// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";

/** LayerScale: escala aprendida por dimensión, inicializada en 1. */
export class LayerScale {
    readonly gamma: Tensor;
    readonly #size: number;

    constructor(size: number) {
        if (!Number.isInteger(size) || size <= 0) {
            throw new Error(`LayerScale necesita una dimensión positiva, no ${size}`);
        }
        this.gamma = Tensor.ones([size], { requiresGrad: true });
        this.#size = size;
    }

    forward(input: Tensor): Tensor {
        if (input.rank !== 2 || (input.shape[1] as number) !== this.#size) {
            throw new Error(
                `LayerScale espera [filas, ${this.#size}], no ${JSON.stringify(input.shape)}`,
            );
        }
        const rows = input.shape[0] as number;
        const size = this.#size;
        const output = new Float32Array(rows * size);
        for (let row = 0; row < rows; row++) {
            for (let column = 0; column < size; column++) {
                output[row * size + column] =
                    (input.data[row * size + column] as number) * (this.gamma.data[column] as number);
            }
        }
        const result = new Tensor(output, [rows, size]);
        if (input.requiresGrad || this.gamma.requiresGrad) {
            result.requiresGrad = true;
            result.setGraph([input, this.gamma], (gradient) => {
                const dGamma = new Float32Array(size);
                const dInput = new Float32Array(rows * size);
                for (let row = 0; row < rows; row++) {
                    for (let column = 0; column < size; column++) {
                        const value = gradient.data[row * size + column] as number;
                        dGamma[column] = (dGamma[column] as number) +
                            value * (input.data[row * size + column] as number);
                        dInput[row * size + column] = value * (this.gamma.data[column] as number);
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
