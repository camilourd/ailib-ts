// Teoría: docs/teoria/secuencias/lm-transformer.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { Embedding } from "../nn/embedding.ts";
import { KVCache } from "../nn/kv-cache.ts";
import type { AttentionCache } from "../nn/kv-cache.ts";
import { LayerNorm } from "../nn/layernorm.ts";
import { Linear } from "../nn/linear.ts";
import { crossEntropy } from "../nn/losses.ts";
import { RMSNorm } from "../nn/rmsnorm.ts";
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
    normKind?: "layer" | "rms";
    ffnKind?: "relu" | "swiglu";
    parallel?: boolean;
}

export interface TransformerGenerateOptions {
    temperature?: number;
    topK?: number;
    topP?: number;
    seed?: number;
    rng?: RandomNumberGenerator;
    useCache?: boolean;
}

/** LM byte-level con bloques transformer, RoPE y máscara causal. */
export class TransformerLM {
    readonly embedding: Embedding;
    readonly blocks: readonly TransformerBlock[];
    readonly finalNorm: LayerNorm | RMSNorm;
    readonly output: Linear;
    readonly #vocabularySize: number;
    readonly #modelSize: number;
    readonly #heads: number;
    readonly #hiddenSize: number;

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
        this.#modelSize = modelSize;
        this.#heads = heads;
        this.#hiddenSize = hiddenSize;
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
                    normKind: options.normKind ?? "layer",
                    ffnKind: options.ffnKind ?? "relu",
                    parallel: options.parallel ?? false,
                }),
        );
        this.finalNorm =
            (options.normKind ?? "layer") === "rms"
                ? new RMSNorm(modelSize)
                : new LayerNorm(modelSize);
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
        const sampling: { temperature: number; topK?: number; topP?: number } = {
            temperature,
            ...(options.topK === undefined ? {} : { topK: options.topK }),
            ...(options.topP === undefined ? {} : { topP: options.topP }),
        };
        if (options.useCache) return this.#generateCached(start, length, sampling, rng);
        const sequence = [...start];
        const generated: number[] = [];
        for (let step = 0; step < length; step++) {
            const logits = this.forward(sequence);
            const row = sequence.length - 1;
            const values = logits.data.subarray(
                row * this.#vocabularySize,
                (row + 1) * this.#vocabularySize,
            );
            const next = sampleFromValues(values, sampling, rng);
            generated.push(next);
            sequence.push(next);
        }
        return generated;
    }

    #generateCached(
        start: readonly number[],
        length: number,
        sampling: { temperature: number; topK?: number; topP?: number },
        rng: RandomNumberGenerator,
    ): number[] {
        if (length === 0) return [];
        const cache = new KVCache(this.blocks.length, this.#heads, this.#modelSize / this.#heads);
        let value = this.embedding.row(start[0] as number);
        for (let block = 0; block < this.blocks.length; block++) {
            value = (this.blocks[block] as TransformerBlock).forwardCached(
                value,
                cache.layers[block] as AttentionCache,
                0,
            );
        }
        for (let position = 1; position < start.length; position++) {
            value = this.embedding.row(start[position] as number);
            for (let block = 0; block < this.blocks.length; block++) {
                value = (this.blocks[block] as TransformerBlock).forwardCached(
                    value,
                    cache.layers[block] as AttentionCache,
                    position,
                );
            }
        }

        const generated: number[] = [];
        for (let step = 0; step < length; step++) {
            const logits = this.output.forward(this.finalNorm.forward(value));
            const next = sampleFromValues(logits.data, sampling, rng);
            generated.push(next);
            if (step + 1 < length) {
                const position = start.length + step;
                value = this.embedding.row(next);
                for (let block = 0; block < this.blocks.length; block++) {
                    value = (this.blocks[block] as TransformerBlock).forwardCached(
                        value,
                        cache.layers[block] as AttentionCache,
                        position,
                    );
                }
            }
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

    get architecture(): {
        modelSize: number;
        heads: number;
        blocks: number;
        hiddenSize: number;
        vocabularySize: number;
    } {
        return {
            modelSize: this.#modelSize,
            heads: this.#heads,
            blocks: this.blocks.length,
            hiddenSize: this.#hiddenSize,
            vocabularySize: this.#vocabularySize,
        };
    }

    parameterCount(): number {
        return this.parameters().reduce((total, parameter) => total + parameter.size, 0);
    }

    /** Nombres estables para guardar y reanudar checkpoints. */
    namedParameters(): Record<string, Tensor> {
        const named: Record<string, Tensor> = {
            "embedding.weights": this.embedding.weights,
        };
        this.blocks.forEach((block, blockIndex) => {
            block.parameters().forEach((parameter, position) => {
                named[`block.${blockIndex}.${position}`] = parameter;
            });
        });
        named["finalNorm.gamma"] = this.finalNorm.gamma;
        if ("beta" in this.finalNorm) named["finalNorm.beta"] = this.finalNorm.beta;
        named["output.weights"] = this.output.weights;
        named["output.bias"] = this.output.bias;
        return named;
    }

    /** Copia parámetros por nombre; exige el conjunto completo y formas compatibles. */
    loadParameters(parameters: Record<string, Tensor>): void {
        const named = this.namedParameters();
        const expected = Object.keys(named).sort();
        const actual = Object.keys(parameters).sort();
        if (expected.join("\u0000") !== actual.join("\u0000")) {
            throw new Error("el checkpoint no coincide con la arquitectura del modelo");
        }
        for (const name of expected) {
            const source = parameters[name] as Tensor;
            const target = named[name] as Tensor;
            if (source.size !== target.size) {
                throw new Error(`«${name}» tiene ${source.size} valores, se esperaban ${target.size}`);
            }
            target.data.set(source.data);
        }
    }
}
