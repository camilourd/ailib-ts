// Teoría: docs/teoria/evolucion/haea.md

import type { RandomNumberGenerator } from "../core/random/index.ts";
import type { Genome, Variation } from "./types.ts";

/** Normal estándar por Box-Muller, a partir de dos uniformes. */
export function standardNormal(rng: RandomNumberGenerator): number {
    const first = 1 - rng.next();
    const second = rng.next();
    return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

export interface GaussianMutationOptions {
    sigma?: number;
}

export class GaussianMutation implements Variation {
    readonly name = "mutacion-gaussiana";
    readonly arity = 1;
    readonly #sigma: number;

    constructor(options: GaussianMutationOptions = {}) {
        const sigma = options.sigma ?? 0.5;
        if (!Number.isFinite(sigma) || sigma <= 0) {
            throw new RangeError(`sigma debe ser positivo, no ${sigma}`);
        }
        this.#sigma = sigma;
    }

    apply(parents: readonly Genome[], rng: RandomNumberGenerator): Genome {
        const parent = parents[0];
        if (!parent) throw new Error("la mutación necesita un padre");
        const child = new Float32Array(parent.length);
        for (let index = 0; index < parent.length; index++) {
            child[index] = (parent[index] as number) + this.#sigma * standardNormal(rng);
        }
        return child;
    }
}

function twoParents(parents: readonly Genome[]): [Genome, Genome] {
    const first = parents[0];
    const second = parents[1];
    if (!first || !second) throw new Error("el cruce necesita dos padres");
    if (first.length !== second.length) throw new Error("los padres deben tener el mismo tamaño");
    return [first, second];
}

export class UniformCrossover implements Variation {
    readonly name = "cruce-uniforme";
    readonly arity = 2;

    apply(parents: readonly Genome[], rng: RandomNumberGenerator): Genome {
        const [first, second] = twoParents(parents);
        const child = new Float32Array(first.length);
        for (let index = 0; index < child.length; index++) {
            child[index] = rng.next() < 0.5 ? (first[index] as number) : (second[index] as number);
        }
        return child;
    }
}

export class ArithmeticCrossover implements Variation {
    readonly name = "cruce-aritmetico";
    readonly arity = 2;

    apply(parents: readonly Genome[], rng: RandomNumberGenerator): Genome {
        const [first, second] = twoParents(parents);
        const alpha = rng.next();
        const child = new Float32Array(first.length);
        for (let index = 0; index < child.length; index++) {
            child[index] = alpha * (first[index] as number) + (1 - alpha) * (second[index] as number);
        }
        return child;
    }
}
