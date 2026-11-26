// Teoría: docs/teoria/secuencias/gru-lstm.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { add, matmul, mul, sub, tanh } from "../core/tensor-ops.ts";
import type { RecurrentCell, RecurrentState } from "./recurrent.ts";
import { sigmoidTensor } from "./sigmoid.ts";

export interface GRUOptions {
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

/** GRU: compuertas de actualización y reinicio sobre el estado oculto. */
export class GRU implements RecurrentCell {
    readonly #zInput: Tensor;
    readonly #zHidden: Tensor;
    readonly #zBias: Tensor;
    readonly #rInput: Tensor;
    readonly #rHidden: Tensor;
    readonly #rBias: Tensor;
    readonly #nInput: Tensor;
    readonly #nHidden: Tensor;
    readonly #nBias: Tensor;
    readonly #inputSize: number;
    readonly #hiddenSize: number;

    constructor(inputSize: number, hiddenSize: number, options: GRUOptions = {}) {
        if (!Number.isInteger(inputSize) || inputSize <= 0) {
            throw new Error(`la entrada debe ser positiva, no ${inputSize}`);
        }
        if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
            throw new Error(`el estado oculto debe ser positivo, no ${hiddenSize}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        const inputLimit = Math.sqrt(6 / (inputSize + hiddenSize));
        const hiddenLimit = Math.sqrt(6 / (hiddenSize + hiddenSize));
        this.#zInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#zHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#zBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#rInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#rHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#rBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#nInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#nHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#nBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#inputSize = inputSize;
        this.#hiddenSize = hiddenSize;
    }

    get inputSize(): number {
        return this.#inputSize;
    }

    get hiddenSize(): number {
        return this.#hiddenSize;
    }

    initial(): RecurrentState {
        return { hidden: Tensor.zeros([1, this.#hiddenSize]) };
    }

    advance(input: Tensor, state: RecurrentState): RecurrentState {
        return { hidden: this.step(input, state.hidden) };
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
        const z = sigmoidTensor(
            add(add(matmul(input, this.#zInput), matmul(hidden, this.#zHidden)), this.#zBias),
        );
        const r = sigmoidTensor(
            add(add(matmul(input, this.#rInput), matmul(hidden, this.#rHidden)), this.#rBias),
        );
        const candidate = tanh(
            add(
                add(matmul(input, this.#nInput), matmul(mul(r, hidden), this.#nHidden)),
                this.#nBias,
            ),
        );
        return add(mul(sub(Tensor.ones(z.shape), z), candidate), mul(z, hidden));
    }

    parameters(): readonly Tensor[] {
        return [
            this.#zInput,
            this.#zHidden,
            this.#zBias,
            this.#rInput,
            this.#rHidden,
            this.#rBias,
            this.#nInput,
            this.#nHidden,
            this.#nBias,
        ];
    }
}
