// Teoría: docs/teoria/entrenamiento/sampling.md

import type { RandomNumberGenerator } from "../core/random/index.ts";

export interface SamplingOptions {
    temperature?: number;
    topK?: number;
    topP?: number;
}

function checked(options: number | SamplingOptions): Required<Pick<SamplingOptions, "temperature" | "topP">> & {
    topK?: number;
} {
    const config = typeof options === "number" ? { temperature: options } : options;
    const temperature = config.temperature ?? 1;
    if (!Number.isFinite(temperature) || temperature <= 0) {
        throw new RangeError(`la temperatura debe ser positiva, no ${temperature}`);
    }
    if (
        config.topK !== undefined &&
        (!Number.isInteger(config.topK) || config.topK <= 0)
    ) {
        throw new RangeError(`top-k debe ser un entero positivo, no ${config.topK}`);
    }
    const topP = config.topP ?? 1;
    if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
        throw new RangeError(`top-p debe estar en (0, 1], no ${topP}`);
    }
    return { temperature, topP, ...(config.topK === undefined ? {} : { topK: config.topK }) };
}

/** Muestrea un índice de una fila de logits con temperatura, top-k y top-p. */
export function sampleFromValues(
    values: ArrayLike<number>,
    options: number | SamplingOptions,
    rng: RandomNumberGenerator,
): number {
    const { temperature, topK, topP } = checked(options);
    if (values.length === 0) throw new Error("no se puede muestrear una fila vacía");

    const candidates = Array.from({ length: values.length }, (_, index) => ({
        index,
        logit: values[index] as number,
    })).sort((first, second) => second.logit - first.logit);
    const kept = topK === undefined ? candidates : candidates.slice(0, topK);
    const maximum = (kept[0] as { logit: number }).logit;
    const probabilities = kept.map((candidate) =>
        Math.exp((candidate.logit - maximum) / temperature),
    );
    let total = probabilities.reduce((sum, value) => sum + value, 0);

    if (topP < 1) {
        let cumulative = 0;
        let cutoff = probabilities.length;
        for (let index = 0; index < probabilities.length && cumulative < topP; index++) {
            cumulative += (probabilities[index] as number) / total;
            cutoff = index + 1;
        }
        probabilities.length = cutoff;
        total = probabilities.reduce((sum, value) => sum + value, 0);
    }

    let threshold = rng.next() * total;
    for (let index = 0; index < probabilities.length; index++) {
        threshold -= probabilities[index] as number;
        if (threshold <= 0) return (kept[index] as { index: number }).index;
    }
    return (kept[probabilities.length - 1] as { index: number }).index;
}
