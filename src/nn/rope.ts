// Teoría: docs/teoria/secuencias/rope.md

import { Tensor } from "../core/tensor.ts";

export interface RoPEOptions {
    base?: number;
    offset?: number;
}

/**
 * Rotary Position Embedding: rota cada par de dimensiones por `posición · θ_i`.
 * La norma se conserva y el producto q·k depende de la distancia entre posiciones.
 */
export function rope(tensor: Tensor, options: RoPEOptions = {}): Tensor {
    if (tensor.rank !== 2) throw new Error("RoPE espera una matriz [secuencia, dimensión]");
    const sequence = tensor.shape[0] as number;
    const dimension = tensor.shape[1] as number;
    if (dimension % 2 !== 0) {
        throw new Error(`RoPE necesita una dimensión par, no ${dimension}`);
    }
    const base = options.base ?? 10000;
    if (!Number.isFinite(base) || base <= 0) {
        throw new RangeError(`la base de RoPE debe ser positiva, no ${base}`);
    }
    const offset = options.offset ?? 0;
    if (!Number.isInteger(offset) || offset < 0) {
        throw new RangeError(`el desplazamiento debe ser un entero no negativo, no ${offset}`);
    }

    const half = dimension / 2;
    const cosines = new Float32Array(sequence * half);
    const sines = new Float32Array(sequence * half);
    const output = new Float32Array(sequence * dimension);
    for (let row = 0; row < sequence; row++) {
        const position = offset + row;
        for (let pair = 0; pair < half; pair++) {
            const angle = position * base ** (-2 * pair / dimension);
            const cosine = Math.cos(angle);
            const sine = Math.sin(angle);
            cosines[row * half + pair] = cosine;
            sines[row * half + pair] = sine;
            const first = tensor.data[row * dimension + pair * 2] as number;
            const second = tensor.data[row * dimension + pair * 2 + 1] as number;
            output[row * dimension + pair * 2] = first * cosine - second * sine;
            output[row * dimension + pair * 2 + 1] = first * sine + second * cosine;
        }
    }

    const result = new Tensor(output, [sequence, dimension]);
    if (tensor.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([tensor], (gradient) => {
            const values = new Float32Array(sequence * dimension);
            for (let row = 0; row < sequence; row++) {
                for (let pair = 0; pair < half; pair++) {
                    const cosine = cosines[row * half + pair] as number;
                    const sine = sines[row * half + pair] as number;
                    const first = gradient.data[row * dimension + pair * 2] as number;
                    const second = gradient.data[row * dimension + pair * 2 + 1] as number;
                    values[row * dimension + pair * 2] = first * cosine + second * sine;
                    values[row * dimension + pair * 2 + 1] = -first * sine + second * cosine;
                }
            }
            tensor.addGrad(new Tensor(values, tensor.shape));
        });
    }
    return result;
}
