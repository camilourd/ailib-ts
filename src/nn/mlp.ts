// Teoría: docs/teoria/redes/mlp.md

import { Tensor } from "../core/tensor.ts";
import { relu } from "../core/tensor-ops.ts";
import { Linear } from "./linear.ts";
import type { Model } from "./model.ts";

export interface MLPOptions {
    seed?: number;
}

/** Perceptrón multicapa denso: Linear + ReLU en las capas ocultas, Linear al final. */
export class MLP implements Model {
    readonly #layers: Linear[];

    constructor(sizes: readonly number[], options: MLPOptions = {}) {
        if (sizes.length < 2) {
            throw new Error("un MLP necesita al menos una entrada y una salida");
        }
        for (const size of sizes) {
            if (!Number.isInteger(size) || size <= 0) {
                throw new Error(`tamaño de capa inválido: ${size}`);
            }
        }
        const seed = options.seed ?? 0;
        this.#layers = [];
        for (let index = 0; index < sizes.length - 1; index++) {
            this.#layers.push(
                new Linear(sizes[index] as number, sizes[index + 1] as number, { seed: seed + index }),
            );
        }
    }

    forward(input: Tensor): Tensor {
        let value = input;
        for (let index = 0; index < this.#layers.length; index++) {
            value = (this.#layers[index] as Linear).forward(value);
            if (index < this.#layers.length - 1) value = relu(value);
        }
        return value;
    }

    parameters(): readonly Tensor[] {
        return this.#layers.flatMap((layer) => layer.parameters());
    }
}
