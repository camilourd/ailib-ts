// Teoría: docs/teoria/redes/rnn.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { add, matmul, tanh } from "../core/tensor-ops.ts";

export interface RNNOptions {
    seed?: number;
    rng?: RandomNumberGenerator;
}

function randomMatrix(
    rows: number,
    columns: number,
    rng: RandomNumberGenerator,
    limit: number,
): Tensor {
    const data = new Float32Array(rows * columns);
    for (let index = 0; index < data.length; index++) {
        data[index] = (rng.next() * 2 - 1) * limit;
    }
    return new Tensor(data, [rows, columns], { requiresGrad: true });
}

/** RNN de Elman: h_t = tanh(x_t W_x + h_{t-1} W_h + b). */
export class RNN {
    readonly #inputWeights: Tensor;
    readonly #hiddenWeights: Tensor;
    readonly #bias: Tensor;
    readonly #inputSize: number;
    readonly #hiddenSize: number;

    constructor(inputSize: number, hiddenSize: number, options: RNNOptions = {}) {
        if (!Number.isInteger(inputSize) || inputSize <= 0) {
            throw new Error(`la entrada debe ser positiva, no ${inputSize}`);
        }
        if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
            throw new Error(`el estado oculto debe ser positivo, no ${hiddenSize}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        this.#inputWeights = randomMatrix(
            inputSize,
            hiddenSize,
            rng,
            Math.sqrt(6 / (inputSize + hiddenSize)),
        );
        this.#hiddenWeights = randomMatrix(
            hiddenSize,
            hiddenSize,
            rng,
            Math.sqrt(6 / (hiddenSize + hiddenSize)),
        );
        this.#bias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#inputSize = inputSize;
        this.#hiddenSize = hiddenSize;
    }

    get inputSize(): number {
        return this.#inputSize;
    }

    get hiddenSize(): number {
        return this.#hiddenSize;
    }

    initialHidden(): Tensor {
        return Tensor.zeros([1, this.#hiddenSize]);
    }

    step(input: Tensor, hidden: Tensor): Tensor {
        if (input.rank !== 2 || input.shape[0] !== 1 || input.shape[1] !== this.#inputSize) {
            throw new Error(
                `la entrada debe ser [1, ${this.#inputSize}], no ${JSON.stringify(input.shape)}`,
            );
        }
        if (hidden.rank !== 2 || hidden.shape[0] !== 1 || hidden.shape[1] !== this.#hiddenSize) {
            throw new Error(
                `el estado debe ser [1, ${this.#hiddenSize}], no ${JSON.stringify(hidden.shape)}`,
            );
        }
        const inputPart = matmul(input, this.#inputWeights);
        const hiddenPart = add(matmul(hidden, this.#hiddenWeights), this.#bias);
        return tanh(add(inputPart, hiddenPart));
    }

    forward(inputs: readonly Tensor[]): Tensor[] {
        const states: Tensor[] = [];
        let hidden = this.initialHidden();
        for (const input of inputs) {
            hidden = this.step(input, hidden);
            states.push(hidden);
        }
        return states;
    }

    parameters(): readonly Tensor[] {
        return [this.#inputWeights, this.#hiddenWeights, this.#bias];
    }
}
