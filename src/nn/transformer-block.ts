// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { add } from "../core/tensor-ops.ts";
import { FeedForward } from "./feedforward.ts";
import { LayerNorm } from "./layernorm.ts";
import { MultiHeadAttention } from "./multihead.ts";

export interface TransformerBlockOptions {
    hiddenSize?: number;
    seed?: number;
    rope?: boolean;
}

export interface TransformerBlockResult {
    output: Tensor;
    weights: Tensor[];
}

/**
 * Bloque transformer pre-norm:
 *   x = x + MultiHead(LayerNorm(x))
 *   y = x + FFN(LayerNorm(x))
 */
export class TransformerBlock {
    readonly #norm1: LayerNorm;
    readonly #attention: MultiHeadAttention;
    readonly #norm2: LayerNorm;
    readonly #ffn: FeedForward;

    constructor(modelSize: number, heads: number, options: TransformerBlockOptions = {}) {
        const hiddenSize = options.hiddenSize ?? modelSize * 4;
        const seed = options.seed ?? 0;
        this.#norm1 = new LayerNorm(modelSize);
        this.#attention = new MultiHeadAttention(modelSize, heads, {
            seed,
            rope: options.rope ?? false,
        });
        this.#norm2 = new LayerNorm(modelSize);
        this.#ffn = new FeedForward(modelSize, hiddenSize, { seed: seed + 4 });
    }

    forward(input: Tensor, options: { causal?: boolean } = {}): TransformerBlockResult {
        const attention = this.#attention.forward(this.#norm1.forward(input), {
            causal: options.causal ?? true,
        });
        const first = add(input, attention.output);
        const output = add(first, this.#ffn.forward(this.#norm2.forward(first)));
        return { output, weights: attention.weights };
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.#norm1.parameters(),
            ...this.#attention.parameters(),
            ...this.#norm2.parameters(),
            ...this.#ffn.parameters(),
        ];
    }
}
