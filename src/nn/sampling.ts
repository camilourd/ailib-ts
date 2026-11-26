// Teoría: docs/teoria/secuencias/lm-transformer.md

import type { RandomNumberGenerator } from "../core/random/index.ts";

/** Muestrea un índice de una fila de logits con temperatura. */
export function sampleFromValues(
    values: ArrayLike<number>,
    temperature: number,
    rng: RandomNumberGenerator,
): number {
    if (!Number.isFinite(temperature) || temperature <= 0) {
        throw new RangeError(`la temperatura debe ser positiva, no ${temperature}`);
    }
    if (values.length === 0) throw new Error("no se puede muestrear una fila vacía");
    let maximum = -Infinity;
    for (let index = 0; index < values.length; index++) {
        const value = values[index] as number;
        if (value > maximum) maximum = value;
    }
    let total = 0;
    const probabilities = new Float32Array(values.length);
    for (let index = 0; index < values.length; index++) {
        const probability = Math.exp(((values[index] as number) - maximum) / temperature);
        probabilities[index] = probability;
        total += probability;
    }
    let threshold = rng.next() * total;
    for (let index = 0; index < probabilities.length; index++) {
        threshold -= probabilities[index] as number;
        if (threshold <= 0) return index;
    }
    return probabilities.length - 1;
}
