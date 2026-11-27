// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { add } from "../core/tensor-ops.ts";
import { Dropout } from "./dropout.ts";
import { FeedForward } from "./feedforward.ts";
import type { AttentionCache } from "./kv-cache.ts";
import { LayerNorm } from "./layernorm.ts";
import { LayerScale } from "./layerscale.ts";
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
    dropout?: number;
    layerScale?: boolean;
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
    readonly #dropout: Dropout | null;
    readonly #attentionScale: LayerScale | null;
    readonly #ffnScale: LayerScale | null;

    constructor(modelSize: number, heads: number, options: TransformerBlockOptions = {}) {
        const hiddenSize = options.hiddenSize ?? modelSize * 4;
        const seed = options.seed ?? 0;
        const normKind = options.normKind ?? "layer";
        const ffnKind = options.ffnKind ?? "relu";
        this.#parallel = options.parallel ?? false;
        const dropout = options.dropout ?? 0;
        this.#dropout =
            dropout > 0 ? new Dropout({ rate: dropout, seed: seed + 9 }) : null;
        this.#attentionScale = options.layerScale ? new LayerScale(modelSize) : null;
        this.#ffnScale = options.layerScale ? new LayerScale(modelSize) : null;
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

    forward(
        input: Tensor,
        options: { causal?: boolean; training?: boolean } = {},
    ): TransformerBlockResult {
        const training = options.training ?? false;
        if (this.#parallel) {
            let output = input;
            let weights: Tensor[] = [];
            if (this.#attention) {
                const attention = this.#attention.forward(this.#norm1.forward(input), {
                    causal: options.causal ?? true,
                });
                let value = attention.output;
                if (this.#dropout) value = this.#dropout.forward(value, { training });
                if (this.#attentionScale) value = this.#attentionScale.forward(value);
                output = add(output, value);
                weights = attention.weights;
            }
            if (this.#ffn) {
                let value = this.#ffn.forward(this.#norm2.forward(input));
                if (this.#dropout) value = this.#dropout.forward(value, { training });
                if (this.#ffnScale) value = this.#ffnScale.forward(value);
                output = add(output, value);
            }
            return { output, weights };
        }
        let first = input;
        let weights: Tensor[] = [];
        if (this.#attention) {
            const attention = this.#attention.forward(this.#norm1.forward(input), {
                causal: options.causal ?? true,
            });
            let value = attention.output;
            if (this.#dropout) value = this.#dropout.forward(value, { training });
            if (this.#attentionScale) value = this.#attentionScale.forward(value);
            first = add(input, value);
            weights = attention.weights;
        }
        if (!this.#ffn) return { output: first, weights };
        let value = this.#ffn.forward(this.#norm2.forward(first));
        if (this.#dropout) value = this.#dropout.forward(value, { training });
        if (this.#ffnScale) value = this.#ffnScale.forward(value);
        return { output: add(first, value), weights };
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
            if (this.#attentionScale) parameters.push(...this.#attentionScale.parameters());
        }
        if (this.#ffn) {
            parameters.push(...this.#norm2.parameters(), ...this.#ffn.parameters());
            if (this.#ffnScale) parameters.push(...this.#ffnScale.parameters());
        }
        return parameters;
    }

    /** Deja las ramas residuales en cero para insertar el bloque sin cambiar la función. */
    zeroResidualOutput(): void {
        this.#attention?.zeroOutput();
        this.#ffn?.zeroOutput();
        this.#attentionScale?.zero();
        this.#ffnScale?.zero();
    }
}
