// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { add, exp, mul, sliceColumns, sum } from "../core/tensor-ops.ts";
import { FeedForward } from "./feedforward.ts";
import { Linear } from "./linear.ts";

export interface MoEOptions {
    experts?: number;
    topK?: number;
    seed?: number;
}

/** Mezcla de expertos top-k con router aprendido sobre el FFN. */
export class MoEFFN {
    readonly #experts: FeedForward[];
    readonly #router: Linear;
    readonly #topK: number;
    #lastUsage: number[];

    constructor(modelSize: number, hiddenSize: number, options: MoEOptions = {}) {
        const experts = options.experts ?? 4;
        const topK = options.topK ?? 1;
        if (!Number.isInteger(experts) || experts <= 0) {
            throw new Error(`número de expertos inválido: ${experts}`);
        }
        if (!Number.isInteger(topK) || topK <= 0 || topK > experts) {
            throw new Error(`top-k inválido: ${topK} para ${experts} expertos`);
        }
        const seed = options.seed ?? 0;
        this.#experts = Array.from(
            { length: experts },
            (_, index) => new FeedForward(modelSize, hiddenSize, { seed: seed + index }),
        );
        this.#router = new Linear(modelSize, experts, { seed: seed + 100 });
        this.#topK = topK;
        this.#lastUsage = new Array(experts).fill(0);
    }

    forward(input: Tensor): Tensor {
        const rows = input.shape[0] as number;
        const experts = this.#experts.length;
        const logits = this.#router.forward(input);
        const probabilities = exp(logits);
        const totals = sum(probabilities, 1);
        const inverse = new Tensor(
            Float32Array.from({ length: rows }, (_, row) => 1 / (totals.data[row] as number)),
            [rows],
        );
        const gates = mul(probabilities, inverse.reshape([rows, 1]));

        const mask = new Float32Array(rows * experts);
        this.#lastUsage = new Array(experts).fill(0);
        for (let row = 0; row < rows; row++) {
            const selected: number[] = [];
            for (let expert = 0; expert < experts; expert++) selected.push(expert);
            selected.sort(
                (left, right) =>
                    (logits.data[row * experts + right] as number) -
                    (logits.data[row * experts + left] as number),
            );
            for (const expert of selected.slice(0, this.#topK)) {
                mask[row * experts + expert] = 1;
                this.#lastUsage[expert] = (this.#lastUsage[expert] as number) + 1;
            }
        }
        const maskedGates = mul(gates, new Tensor(mask, [rows, experts]));

        let output: Tensor | null = null;
        for (let expert = 0; expert < experts; expert++) {
            const gate = sliceColumns(maskedGates, expert, expert + 1);
            const value = mul(this.#experts[expert]!.forward(input), gate);
            output = output === null ? value : add(output, value);
        }
        if (output === null) throw new Error("el MoE no tiene expertos");
        return output;
    }

    /** Uso del último forward, por experto. */
    usage(): readonly number[] {
        return this.#lastUsage;
    }

    parameters(): readonly Tensor[] {
        return [
            ...this.#experts.flatMap((expert) => expert.parameters()),
            ...this.#router.parameters(),
        ];
    }
}
