// Teoría: docs/teoria/redes/dropout.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";

export interface DropoutOptions {
    rate?: number;
    seed?: number;
    rng?: RandomNumberGenerator;
}

/** Dropout invertido: apaga con probabilidad `rate` y escala por 1/(1-rate). */
export class Dropout {
    readonly #rate: number;
    readonly #rng: RandomNumberGenerator;

    constructor(options: DropoutOptions = {}) {
        const rate = options.rate ?? 0.1;
        if (!Number.isFinite(rate) || rate < 0 || rate >= 1) {
            throw new RangeError(`dropout debe estar en [0, 1), no ${rate}`);
        }
        this.#rate = rate;
        this.#rng = options.rng ?? createRng(options.seed ?? 0);
    }

    forward(input: Tensor, options: { training?: boolean } = {}): Tensor {
        if (!(options.training ?? false) || this.#rate === 0) return input;
        const scale = 1 / (1 - this.#rate);
        const mask = new Float32Array(input.size);
        const output = new Float32Array(input.size);
        for (let index = 0; index < input.size; index++) {
            const keep = this.#rng.next() >= this.#rate ? 1 : 0;
            mask[index] = keep * scale;
            output[index] = (input.data[index] as number) * (mask[index] as number);
        }
        const result = new Tensor(output, input.shape);
        if (input.requiresGrad) {
            result.requiresGrad = true;
            result.setGraph([input], (gradient) => {
                const values = new Float32Array(input.size);
                for (let index = 0; index < input.size; index++) {
                    values[index] = (gradient.data[index] as number) * (mask[index] as number);
                }
                input.addGrad(new Tensor(values, input.shape));
            });
        }
        return result;
    }

    parameters(): readonly Tensor[] {
        return [];
    }
}
