// Teoría: docs/teoria/secuencias/multi-head.md

import { Tensor } from "../core/tensor.ts";
import { concatColumns, sliceColumns } from "../core/tensor-ops.ts";
import { scaledDotProductAttention } from "./attention.ts";
import { appendRows } from "./kv-cache.ts";
import type { AttentionCache } from "./kv-cache.ts";
import { Linear } from "./linear.ts";
import { rope } from "./rope.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";

export interface MultiHeadAttentionOptions {
    seed?: number;
    rng?: RandomNumberGenerator;
    rope?: boolean;
}

export interface MultiHeadAttentionResult {
    output: Tensor;
    weights: Tensor[];
}

/** Atención multi-cabeza: proyecta Q/K/V, parte en cabezas, atiende y vuelve a juntar. */
export class MultiHeadAttention {
    readonly #query: Linear;
    readonly #key: Linear;
    readonly #value: Linear;
    readonly #output: Linear;
    readonly #heads: number;
    readonly #headSize: number;
    readonly #modelSize: number;
    readonly #rope: boolean;

    constructor(modelSize: number, heads: number, options: MultiHeadAttentionOptions = {}) {
        if (!Number.isInteger(modelSize) || modelSize <= 0) {
            throw new Error(`la dimensión del modelo debe ser positiva, no ${modelSize}`);
        }
        if (!Number.isInteger(heads) || heads <= 0 || modelSize % heads !== 0) {
            throw new Error(`las cabezas deben dividir ${modelSize}, no ${heads}`);
        }
        const seed = options.seed ?? 0;
        this.#modelSize = modelSize;
        this.#heads = heads;
        this.#headSize = modelSize / heads;
        this.#rope = options.rope ?? false;
        this.#query = new Linear(modelSize, modelSize, { seed });
        this.#key = new Linear(modelSize, modelSize, { seed: seed + 1 });
        this.#value = new Linear(modelSize, modelSize, { seed: seed + 2 });
        this.#output = new Linear(modelSize, modelSize, { seed: seed + 3 });
    }

    get heads(): number {
        return this.#heads;
    }

    get headSize(): number {
        return this.#headSize;
    }

    forward(input: Tensor, options: { causal?: boolean } = {}): MultiHeadAttentionResult {
        if (input.rank !== 2 || (input.shape[1] as number) !== this.#modelSize) {
            throw new Error(
                `la entrada debe ser [secuencia, ${this.#modelSize}], no ${JSON.stringify(input.shape)}`,
            );
        }
        const queries = this.#query.forward(input);
        const keys = this.#key.forward(input);
        const values = this.#value.forward(input);
        const outputs: Tensor[] = [];
        const weights: Tensor[] = [];
        for (let head = 0; head < this.#heads; head++) {
            const start = head * this.#headSize;
            const end = start + this.#headSize;
            let queriesHead = sliceColumns(queries, start, end);
            let keysHead = sliceColumns(keys, start, end);
            if (this.#rope) {
                queriesHead = rope(queriesHead);
                keysHead = rope(keysHead);
            }
            const result = scaledDotProductAttention(
                queriesHead,
                keysHead,
                sliceColumns(values, start, end),
                { causal: options.causal ?? true },
            );
            outputs.push(result.output);
            weights.push(result.weights);
        }
        return {
            output: this.#output.forward(concatColumns(outputs)),
            weights,
        };
    }

    /** Un token nuevo contra la caché de K/V; el pasado no se recalcula. */
    forwardStep(input: Tensor, cache: AttentionCache, position: number): Tensor {
        if (input.rank !== 2 || input.shape[0] !== 1 || input.shape[1] !== this.#modelSize) {
            throw new Error(
                `la entrada debe ser [1, ${this.#modelSize}], no ${JSON.stringify(input.shape)}`,
            );
        }
        if (!Number.isInteger(position) || position < 0) {
            throw new RangeError(`posición inválida: ${position}`);
        }
        if (cache.keys.length !== this.#heads) {
            throw new Error(`la caché tiene ${cache.keys.length} cabezas, se esperaban ${this.#heads}`);
        }
        for (let head = 0; head < this.#heads; head++) {
            if ((cache.keys[head]?.shape[0] as number) !== position) {
                throw new Error(`la caché no está en la posición ${position}`);
            }
        }
        const queries = this.#query.forward(input);
        const keys = this.#key.forward(input);
        const values = this.#value.forward(input);
        const outputs: Tensor[] = [];
        for (let head = 0; head < this.#heads; head++) {
            const start = head * this.#headSize;
            const end = start + this.#headSize;
            let query = sliceColumns(queries, start, end);
            let key = sliceColumns(keys, start, end);
            if (this.#rope) {
                query = rope(query, { offset: position });
                key = rope(key, { offset: position });
            }
            const previousKeys = cache.keys[head] as Tensor;
            const previousValues = cache.values[head] as Tensor;
            cache.keys[head] = appendRows(previousKeys, key);
            cache.values[head] = appendRows(previousValues, sliceColumns(values, start, end));
            const result = scaledDotProductAttention(
                query,
                cache.keys[head] as Tensor,
                cache.values[head] as Tensor,
                { causal: false },
            );
            outputs.push(result.output);
        }
        return this.#output.forward(concatColumns(outputs));
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.#query.parameters(),
            ...this.#key.parameters(),
            ...this.#value.parameters(),
            ...this.#output.parameters(),
        ];
    }
}
