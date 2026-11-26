// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { relu } from "../core/tensor-ops.ts";
import { Tensor } from "../core/tensor.ts";
import { Linear } from "./linear.ts";

export interface FeedForwardOptions {
    seed?: number;
}

/** FFN del transformer: Linear → ReLU → Linear. */
export class FeedForward {
    readonly #first: Linear;
    readonly #second: Linear;

    constructor(modelSize: number, hiddenSize: number, options: FeedForwardOptions = {}) {
        const seed = options.seed ?? 0;
        this.#first = new Linear(modelSize, hiddenSize, { seed });
        this.#second = new Linear(hiddenSize, modelSize, { seed: seed + 1 });
    }

    forward(input: Tensor): Tensor {
        return this.#second.forward(relu(this.#first.forward(input)));
    }

    parameters(): readonly Tensor[] {
        return [...this.#first.parameters(), ...this.#second.parameters()];
    }
}
