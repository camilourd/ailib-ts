// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { add } from "../core/tensor-ops.ts";
import { FeedForward } from "./feedforward.ts";
import type { AttentionCache } from "./kv-cache.ts";
import { LayerNorm } from "./layernorm.ts";
import { MultiHeadAttention } from "./multihead.ts";

export interface TransformerBlockOptions {
    hiddenSize?: number;
    seed?: number;
    rope?: boolean;
    attention?: boolean;
    ffn?: boolean;
}

export interface TransformerBlockResult {
    output: Tensor;
    weights: Tensor[];
}

/**
 * Bloque transformer pre-norm:
 *   x = x + MultiHead(LayerNorm(x))
 *   y = x + FFN(LayerNorm(x))
 * Las piezas se pueden apagar para la ablación.
 */
export class TransformerBlock {
    readonly #norm1: LayerNorm;
    readonly #attention: MultiHeadAttention | null;
    readonly #norm2: LayerNorm;
    readonly #ffn: FeedForward | null;

    constructor(modelSize: number, heads: number, options: TransformerBlockOptions = {}) {
        const hiddenSize = options.hiddenSize ?? modelSize * 4;
        const seed = options.seed ?? 0;
        this.#norm1 = new LayerNorm(modelSize);
        this.#attention =
            options.attention ?? true
                ? new MultiHeadAttention(modelSize, heads, {
                      seed,
                      rope: options.rope ?? false,
                  })
                : null;
        this.#norm2 = new LayerNorm(modelSize);
        this.#ffn =
            options.ffn ?? true
                ? new FeedForward(modelSize, hiddenSize, { seed: seed + 4 })
                : null;
    }

    forward(input: Tensor, options: { causal?: boolean } = {}): TransformerBlockResult {
        let first = input;
        let weights: Tensor[] = [];
        if (this.#attention) {
            const attention = this.#attention.forward(this.#norm1.forward(input), {
                causal: options.causal ?? true,
            });
            first = add(input, attention.output);
            weights = attention.weights;
        }
        const output = this.#ffn
            ? add(first, this.#ffn.forward(this.#norm2.forward(first)))
            : first;
        return { output, weights };
    }

    /** Un token nuevo con caché de atención; requiere que la atención esté activa. */
    forwardCached(input: Tensor, cache: AttentionCache, position: number): Tensor {
        if (!this.#attention) throw new Error("el bloque no tiene atención");
        const first = add(
            input,
            this.#attention.forwardStep(this.#norm1.forward(input), cache, position),
        );
        return this.#ffn
            ? add(first, this.#ffn.forward(this.#norm2.forward(first)))
            : first;
    }

    parameters(): readonly Tensor[] {
        const parameters: Tensor[] = [];
        if (this.#attention) {
            parameters.push(...this.#norm1.parameters(), ...this.#attention.parameters());
        }
        if (this.#ffn) {
            parameters.push(...this.#norm2.parameters(), ...this.#ffn.parameters());
        }
        return parameters;
    }
}
