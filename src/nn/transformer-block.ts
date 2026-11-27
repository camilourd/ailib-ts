// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { add } from "../core/tensor-ops.ts";
import { FeedForward } from "./feedforward.ts";
import type { AttentionCache } from "./kv-cache.ts";
import { LayerNorm } from "./layernorm.ts";
import { MultiHeadAttention } from "./multihead.ts";
import { RMSNorm } from "./rmsnorm.ts";
import { SwiGLU } from "./swiglu.ts";

export interface TransformerBlockOptions {
    hiddenSize?: number;
    seed?: number;
    rope?: boolean;
    attention?: boolean;
    ffn?: boolean;
    normKind?: "layer" | "rms";
    ffnKind?: "relu" | "swiglu";
    parallel?: boolean;
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
    readonly #norm1: LayerNorm | RMSNorm;
    readonly #attention: MultiHeadAttention | null;
    readonly #norm2: LayerNorm | RMSNorm;
    readonly #ffn: FeedForward | SwiGLU | null;
    readonly #parallel: boolean;

    constructor(modelSize: number, heads: number, options: TransformerBlockOptions = {}) {
        const hiddenSize = options.hiddenSize ?? modelSize * 4;
        const seed = options.seed ?? 0;
        const normKind = options.normKind ?? "layer";
        const ffnKind = options.ffnKind ?? "relu";
        this.#parallel = options.parallel ?? false;
        this.#norm1 = normKind === "rms" ? new RMSNorm(modelSize) : new LayerNorm(modelSize);
        this.#attention =
            options.attention ?? true
                ? new MultiHeadAttention(modelSize, heads, {
                      seed,
                      rope: options.rope ?? false,
                  })
                : null;
        this.#norm2 = normKind === "rms" ? new RMSNorm(modelSize) : new LayerNorm(modelSize);
        this.#ffn =
            options.ffn ?? true
                ? ffnKind === "swiglu"
                    ? new SwiGLU(modelSize, hiddenSize, { seed: seed + 4 })
                    : new FeedForward(modelSize, hiddenSize, { seed: seed + 4 })
                : null;
    }

    forward(input: Tensor, options: { causal?: boolean } = {}): TransformerBlockResult {
        if (this.#parallel) {
            let output = input;
            let weights: Tensor[] = [];
            if (this.#attention) {
                const attention = this.#attention.forward(this.#norm1.forward(input), {
                    causal: options.causal ?? true,
                });
                output = add(output, attention.output);
                weights = attention.weights;
            }
            if (this.#ffn) output = add(output, this.#ffn.forward(this.#norm2.forward(input)));
            return { output, weights };
        }
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
        if (this.#parallel) {
            let output = add(
                input,
                this.#attention.forwardStep(this.#norm1.forward(input), cache, position),
            );
            if (this.#ffn) output = add(output, this.#ffn.forward(this.#norm2.forward(input)));
            return output;
        }
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
