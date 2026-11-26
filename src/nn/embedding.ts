// Teoría: docs/teoria/redes/embeddings.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";

export interface EmbeddingOptions {
    seed?: number;
    rng?: RandomNumberGenerator;
}

/** Tabla de embeddings: cada token se busca como una fila de `weights`. */
export class Embedding {
    readonly weights: Tensor;
    readonly #vocabularySize: number;
    readonly #dimension: number;

    constructor(vocabularySize: number, dimension: number, options: EmbeddingOptions = {}) {
        if (!Number.isInteger(vocabularySize) || vocabularySize <= 0) {
            throw new Error(`el vocabulario debe ser positivo, no ${vocabularySize}`);
        }
        if (!Number.isInteger(dimension) || dimension <= 0) {
            throw new Error(`la dimensión debe ser positiva, no ${dimension}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        const limit = 1 / Math.sqrt(dimension);
        const data = new Float32Array(vocabularySize * dimension);
        for (let index = 0; index < data.length; index++) {
            data[index] = (rng.next() * 2 - 1) * limit;
        }
        this.weights = new Tensor(data, [vocabularySize, dimension], { requiresGrad: true });
        this.#vocabularySize = vocabularySize;
        this.#dimension = dimension;
    }

    get vocabularySize(): number {
        return this.#vocabularySize;
    }

    get dimension(): number {
        return this.#dimension;
    }

    /** Devuelve una fila por índice; el backward reparte el gradiente en esas filas. */
    forward(indices: readonly number[]): Tensor {
        const rows = indices.length;
        const result = new Tensor(new Float32Array(rows * this.#dimension), [
            rows,
            this.#dimension,
        ]);
        for (let row = 0; row < rows; row++) {
            const token = this.#validate(indices[row] as number, row);
            result.data.set(
                this.weights.data.subarray(
                    token * this.#dimension,
                    (token + 1) * this.#dimension,
                ),
                row * this.#dimension,
            );
        }
        if (this.weights.requiresGrad) {
            result.requiresGrad = true;
            result.setGraph([this.weights], (gradient) => {
                const values = new Float32Array(this.weights.size);
                for (let row = 0; row < rows; row++) {
                    const token = indices[row] as number;
                    for (let column = 0; column < this.#dimension; column++) {
                        const position = token * this.#dimension + column;
                        values[position] =
                            (values[position] as number) +
                            (gradient.data[row * this.#dimension + column] as number);
                    }
                }
                this.weights.addGrad(new Tensor(values, this.weights.shape));
            });
        }
        return result;
    }

    row(index: number): Tensor {
        return this.forward([index]);
    }

    parameters(): readonly Tensor[] {
        return [this.weights];
    }

    #validate(index: number, row: number): number {
        if (!Number.isInteger(index) || index < 0 || index >= this.#vocabularySize) {
            throw new RangeError(
                `índice ${index} fuera de [0, ${this.#vocabularySize}) en la fila ${row}`,
            );
        }
        return index;
    }
}
