// Teoría: docs/teoria/evolucion/haea.md

import type { RandomNumberGenerator } from "../core/random/index.ts";

export type Genome = Float32Array;

export type Fitness = (genome: Genome) => number;

export interface Individual {
    genome: Genome;
    fitness: number;
}

export interface Variation {
    readonly name: string;
    readonly arity: number;
    apply(parents: readonly Genome[], rng: RandomNumberGenerator): Genome;
}
