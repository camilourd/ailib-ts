// Teoría: docs/teoria/secuencias/lm-transformer.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { Embedding } from "../nn/embedding.ts";
import { LayerNorm } from "../nn/layernorm.ts";
import { Linear } from "../nn/linear.ts";
import { crossEntropy } from "../nn/losses.ts";
import { sampleFromValues } from "../nn/sampling.ts";
import { TransformerBlock } from "../nn/transformer-block.ts";

export interface TransformerLMOptions {
    modelSize?: number;
    heads?: number;
    blocks?: number;
    hiddenSize?: number;
    seed?: number;
    rope?: boolean;
    attention?: boolean;
    ffn?: boolean;
}

export interface TransformerGenerateOptions {
    temperature?: number;
    seed?: number;
    rng?: RandomNumberGenerator;
}

/** LM byte-level con bloques transformer, RoPE y máscara causal. */
export class TransformerLM {
    readonly embedding: Embedding;
    readonly blocks: readonly TransformerBlock[];
    readonly finalNorm: LayerNorm;
    readonly output: Linear;
    readonly #vocabularySize: number;

    constructor(vocabularySize: number, options: TransformerLMOptions = {}) {
        const modelSize = options.modelSize ?? 32;
        const heads = options.heads ?? 4;
        const blocks = options.blocks ?? 1;
        const hiddenSize = options.hiddenSize ?? modelSize * 4;
        const seed = options.seed ?? 0;
        if (!Number.isInteger(blocks) || blocks <= 0) {
            throw new Error(`el transformer necesita al menos un bloque, no ${blocks}`);
        }
        this.#vocabularySize = vocabularySize;
        this.embedding = new Embedding(vocabularySize, modelSize, { seed });
        this.blocks = Array.from(
            { length: blocks },
            (_, index) =>
                new TransformerBlock(modelSize, heads, {
                    hiddenSize,
                    seed: seed + 10 + index * 10,
                    rope: options.rope ?? true,
                    attention: options.attention ?? true,
                    ffn: options.ffn ?? true,
                }),
        );
        this.finalNorm = new LayerNorm(modelSize);
        this.output = new Linear(modelSize, vocabularySize, { seed: seed + 10 + blocks * 10 });
    }

    forward(tokens: readonly number[]): Tensor {
        if (tokens.length === 0) throw new Error("hace falta al menos un token");
        let value = this.embedding.forward(tokens);
        for (const block of this.blocks) {
            value = block.forward(value).output;
        }
        return this.output.forward(this.finalNorm.forward(value));
    }

    /** Cross-entropy media de predecir cada token a partir del anterior. */
    loss(tokens: readonly number[]): Tensor {
        if (tokens.length < 2) {
            throw new Error("hacen falta al menos dos tokens para predecir");
        }
        return crossEntropy(this.forward(tokens.slice(0, -1)), tokens.slice(1));
    }

    /** Devuelve la continuación generada, sin incluir el arranque. */
    generate(
        start: readonly number[],
        length: number,
        options: TransformerGenerateOptions = {},
    ): number[] {
        if (start.length === 0) throw new Error("hace falta un token de arranque");
        if (!Number.isInteger(length) || length < 0) {
            throw new RangeError(`la longitud debe ser un entero no negativo, no ${length}`);
        }
        const temperature = options.temperature ?? 0.8;
        if (!Number.isFinite(temperature) || temperature <= 0) {
            throw new RangeError(`la temperatura debe ser positiva, no ${temperature}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        const sequence = [...start];
        const generated: number[] = [];
        for (let step = 0; step < length; step++) {
            const logits = this.forward(sequence);
            const row = sequence.length - 1;
            const values = logits.data.subarray(
                row * this.#vocabularySize,
                (row + 1) * this.#vocabularySize,
            );
            const next = sampleFromValues(values, temperature, rng);
            generated.push(next);
            sequence.push(next);
        }
        return generated;
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.embedding.parameters(),
            ...this.blocks.flatMap((block) => block.parameters()),
            ...this.finalNorm.parameters(),
            ...this.output.parameters(),
        ];
    }
}
