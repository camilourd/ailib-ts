// Teoría: docs/teoria/secuencias/gru-lstm.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { add, matmul, mul, tanh } from "../core/tensor-ops.ts";
import type { RecurrentCell, RecurrentState } from "./recurrent.ts";
import { sigmoidTensor } from "./sigmoid.ts";

export interface LSTMOptions {
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

/** LSTM: compuertas de entrada, olvido y salida, más un estado de celda. */
export class LSTM implements RecurrentCell {
    readonly #iInput: Tensor;
    readonly #iHidden: Tensor;
    readonly #iBias: Tensor;
    readonly #fInput: Tensor;
    readonly #fHidden: Tensor;
    readonly #fBias: Tensor;
    readonly #oInput: Tensor;
    readonly #oHidden: Tensor;
    readonly #oBias: Tensor;
    readonly #gInput: Tensor;
    readonly #gHidden: Tensor;
    readonly #gBias: Tensor;
    readonly #inputSize: number;
    readonly #hiddenSize: number;

    constructor(inputSize: number, hiddenSize: number, options: LSTMOptions = {}) {
        if (!Number.isInteger(inputSize) || inputSize <= 0) {
            throw new Error(`la entrada debe ser positiva, no ${inputSize}`);
        }
        if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
            throw new Error(`el estado oculto debe ser positivo, no ${hiddenSize}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        const inputLimit = Math.sqrt(6 / (inputSize + hiddenSize));
        const hiddenLimit = Math.sqrt(6 / (hiddenSize + hiddenSize));
        this.#iInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#iHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#iBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#fInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#fHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#fBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#oInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#oHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#oBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
        this.#gInput = randomMatrix(inputSize, hiddenSize, rng, inputLimit);
        this.#gHidden = randomMatrix(hiddenSize, hiddenSize, rng, hiddenLimit);
        this.#gBias = Tensor.zeros([hiddenSize], { requiresGrad: true });
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
        return {
            hidden: Tensor.zeros([1, this.#hiddenSize]),
            cell: Tensor.zeros([1, this.#hiddenSize]),
        };
    }

    advance(input: Tensor, state: RecurrentState): RecurrentState {
        const cell = state.cell ?? Tensor.zeros([1, this.#hiddenSize]);
        return this.step(input, state.hidden, cell);
    }

    step(input: Tensor, hidden: Tensor, cell: Tensor): RecurrentState {
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
        if (cell.rank !== 2 || cell.shape[0] !== 1 || cell.shape[1] !== this.#hiddenSize) {
            throw new Error(
                `la celda debe ser [1, ${this.#hiddenSize}], no ${JSON.stringify(cell.shape)}`,
            );
        }
        const inputGate = sigmoidTensor(
            add(add(matmul(input, this.#iInput), matmul(hidden, this.#iHidden)), this.#iBias),
        );
        const forgetGate = sigmoidTensor(
            add(add(matmul(input, this.#fInput), matmul(hidden, this.#fHidden)), this.#fBias),
        );
        const outputGate = sigmoidTensor(
            add(add(matmul(input, this.#oInput), matmul(hidden, this.#oHidden)), this.#oBias),
        );
        const candidate = tanh(
            add(add(matmul(input, this.#gInput), matmul(hidden, this.#gHidden)), this.#gBias),
        );
        const newCell = add(mul(forgetGate, cell), mul(inputGate, candidate));
        return {
            hidden: mul(outputGate, tanh(newCell)),
            cell: newCell,
        };
    }

    parameters(): readonly Tensor[] {
        return [
            this.#iInput,
            this.#iHidden,
            this.#iBias,
            this.#fInput,
            this.#fHidden,
            this.#fBias,
            this.#oInput,
            this.#oHidden,
            this.#oBias,
            this.#gInput,
            this.#gHidden,
            this.#gBias,
        ];
    }
}
