// Teoría: docs/teoria/entrenamiento/preentreno.md

import type { Tensor } from "../core/tensor.ts";

/** Recorta la norma global de los gradientes; devuelve la norma antes de recortar. */
export function clipGradients(parameters: readonly Tensor[], maxNorm: number): number {
    if (!Number.isFinite(maxNorm) || maxNorm <= 0) {
        throw new RangeError(`la norma máxima debe ser positiva, no ${maxNorm}`);
    }
    let squares = 0;
    for (const parameter of parameters) {
        const gradient = parameter.grad;
        if (gradient) {
            for (const value of gradient.data) squares += value * value;
        }
    }
    const norm = Math.sqrt(squares);
    if (norm > maxNorm) {
        const scale = maxNorm / norm;
        for (const parameter of parameters) {
            const gradient = parameter.grad;
            if (gradient) {
                for (let index = 0; index < gradient.data.length; index++) {
                    gradient.data[index] = (gradient.data[index] as number) * scale;
                }
            }
        }
    }
    return norm;
}
