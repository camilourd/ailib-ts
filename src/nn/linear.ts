// Teoría: docs/teoria/redes/capa-lineal.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { add, matmul } from "../core/tensor-ops.ts";

export interface LinearOptions {
    seed?: number;
    rng?: RandomNumberGenerator;
}

/** Capa densa: y = xW + b, con W [entrada, salida] y b [salida]. */
export class Linear {
    readonly weights: Tensor;
    readonly bias: Tensor;
    readonly #inputSize: number;
    readonly #outputSize: number;

    constructor(inputSize: number, outputSize: number, options: LinearOptions = {}) {
        if (!Number.isInteger(inputSize) || inputSize <= 0 || !Number.isInteger(outputSize) || outputSize <= 0) {
            throw new Error(`tamaños de capa inválidos: ${inputSize} → ${outputSize}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        // Xavier uniforme: U(-sqrt(6/(in+out)), +sqrt(6/(in+out))).
        const limit = Math.sqrt(6 / (inputSize + outputSize));
        const data = new Float32Array(inputSize * outputSize);
        for (let index = 0; index < data.length; index++) {
            data[index] = (rng.next() * 2 - 1) * limit;
        }
        this.weights = new Tensor(data, [inputSize, outputSize], { requiresGrad: true });
        this.bias = Tensor.zeros([outputSize], { requiresGrad: true });
        this.#inputSize = inputSize;
        this.#outputSize = outputSize;
    }

    get inputSize(): number {
        return this.#inputSize;
    }

    get outputSize(): number {
        return this.#outputSize;
    }

    forward(input: Tensor): Tensor {
        if (input.rank !== 2 || (input.shape[1] as number) !== this.#inputSize) {
            throw new Error(
                `la entrada debe ser [lote, ${this.#inputSize}], no ${JSON.stringify(input.shape)}`,
            );
        }
        return add(matmul(input, this.weights), this.bias);
    }

    parameters(): readonly Tensor[] {
        return [this.weights, this.bias];
    }
}
