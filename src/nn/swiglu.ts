// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { mul } from "../core/tensor-ops.ts";
import { Linear } from "./linear.ts";
import { silu } from "./silu.ts";

export interface SwiGLUOptions {
    seed?: number;
}

/** FFN SwiGLU: Linear → SiLU(gate) · value → Linear. */
export class SwiGLU {
    readonly #gate: Linear;
    readonly #value: Linear;
    readonly #output: Linear;

    constructor(modelSize: number, hiddenSize: number, options: SwiGLUOptions = {}) {
        const seed = options.seed ?? 0;
        this.#gate = new Linear(modelSize, hiddenSize, { seed });
        this.#value = new Linear(modelSize, hiddenSize, { seed: seed + 1 });
        this.#output = new Linear(hiddenSize, modelSize, { seed: seed + 2 });
    }

    forward(input: Tensor): Tensor {
        return this.#output.forward(mul(silu(this.#gate.forward(input)), this.#value.forward(input)));
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.#gate.parameters(),
            ...this.#value.parameters(),
            ...this.#output.parameters(),
        ];
    }

    zeroOutput(): void {
        this.#output.weights.data.fill(0);
        this.#output.bias.data.fill(0);
    }
}
