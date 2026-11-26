// Teoría: docs/teoria/evolucion/haea.md

import type { RandomNumberGenerator } from "../core/random/index.ts";
import type { Individual } from "./types.ts";

/** Torneo con reemplazo: elige `size` individuos al azar y devuelve el de menor fitness. */
export function tournament(
    population: readonly Individual[],
    size: number,
    rng: RandomNumberGenerator,
): Individual {
    if (population.length === 0) throw new Error("no se puede elegir de una población vacía");
    if (!Number.isInteger(size) || size <= 0) {
        throw new RangeError(`el tamaño del torneo debe ser un entero positivo, no ${size}`);
    }
    let best = population[Math.floor(rng.next() * population.length)] as Individual;
    for (let pick = 1; pick < size; pick++) {
        const candidate = population[Math.floor(rng.next() * population.length)] as Individual;
        if (candidate.fitness < best.fitness) best = candidate;
    }
    return best;
}
