// Teoría: docs/teoria/secuencias/lm-caracteres.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { Tensor } from "../core/tensor.ts";
import { add, mul } from "../core/tensor-ops.ts";
import { Embedding } from "../nn/embedding.ts";
import { Linear } from "../nn/linear.ts";
import { crossEntropy } from "../nn/losses.ts";
import { RNN } from "../nn/rnn.ts";

export interface CharLMOptions {
    embeddingSize?: number;
    hiddenSize?: number;
    seed?: number;
}

export interface GenerateOptions {
    temperature?: number;
    seed?: number;
    rng?: RandomNumberGenerator;
}

export interface CharLMForward {
    logits: Tensor[];
    states: Tensor[];
}

export interface CharLMLoss {
    loss: Tensor;
    states: Tensor[];
}

/** LM de caracteres byte-level: Embedding → RNN → Linear sobre el vocabulario. */
export class CharLM {
    readonly embedding: Embedding;
    readonly rnn: RNN;
    readonly output: Linear;

    constructor(vocabularySize: number, options: CharLMOptions = {}) {
        const embeddingSize = options.embeddingSize ?? 16;
        const hiddenSize = options.hiddenSize ?? 32;
        const seed = options.seed ?? 0;
        this.embedding = new Embedding(vocabularySize, embeddingSize, { seed });
        this.rnn = new RNN(embeddingSize, hiddenSize, { seed: seed + 1 });
        this.output = new Linear(hiddenSize, vocabularySize, { seed: seed + 2 });
    }

    forward(tokens: readonly number[]): CharLMForward {
        if (tokens.length === 0) throw new Error("hace falta al menos un token");
        const rows = tokens.map((token) => this.embedding.row(token));
        const states = this.rnn.forward(rows);
        const logits = states.map((state) => this.output.forward(state));
        return { logits, states };
    }

    /** Pérdida media de predecir cada token a partir del anterior. */
    loss(tokens: readonly number[]): Tensor {
        return this.lossAndStates(tokens).loss;
    }

    /** Igual que `loss`, pero devuelve también los estados para estudiar el gradiente. */
    lossAndStates(tokens: readonly number[]): CharLMLoss {
        if (tokens.length < 2) {
            throw new Error("hacen falta al menos dos tokens para predecir");
        }
        const { logits, states } = this.forward(tokens.slice(0, -1));
        let total = crossEntropy(logits[0] as Tensor, [tokens[1] as number]);
        for (let index = 1; index < logits.length; index++) {
            total = add(
                total,
                crossEntropy(logits[index] as Tensor, [tokens[index + 1] as number]),
            );
        }
        return {
            loss: mul(total, Tensor.from([1 / logits.length], [])),
            states,
        };
    }

    /** Devuelve la continuación generada, sin incluir el arranque. */
    generate(start: readonly number[], length: number, options: GenerateOptions = {}): number[] {
        if (start.length === 0) throw new Error("hace falta un token de arranque");
        if (!Number.isInteger(length) || length < 0) {
            throw new RangeError(`la longitud debe ser un entero no negativo, no ${length}`);
        }
        const temperature = options.temperature ?? 0.8;
        if (!Number.isFinite(temperature) || temperature <= 0) {
            throw new RangeError(`la temperatura debe ser positiva, no ${temperature}`);
        }
        const rng = options.rng ?? createRng(options.seed ?? 0);
        let hidden = this.rnn.initialHidden();
        for (const token of start) {
            hidden = this.rnn.step(this.embedding.row(token), hidden);
        }
        const generated: number[] = [];
        for (let step = 0; step < length; step++) {
            const next = sample(this.output.forward(hidden), temperature, rng);
            generated.push(next);
            hidden = this.rnn.step(this.embedding.row(next), hidden);
        }
        return generated;
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.embedding.parameters(),
            ...this.rnn.parameters(),
            ...this.output.parameters(),
        ];
    }
}

function sample(logits: Tensor, temperature: number, rng: RandomNumberGenerator): number {
    const values = [...logits.data];
    let maximum = -Infinity;
    for (const value of values) {
        if (value > maximum) maximum = value;
    }
    let total = 0;
    const probabilities = values.map((value) => {
        const probability = Math.exp((value - maximum) / temperature);
        total += probability;
        return probability;
    });
    let threshold = rng.next() * total;
    for (let index = 0; index < probabilities.length; index++) {
        threshold -= probabilities[index] as number;
        if (threshold <= 0) return index;
    }
    return probabilities.length - 1;
}
