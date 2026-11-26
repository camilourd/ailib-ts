// Teoría: docs/teoria/entrenamiento/inferencia.md

import { Tensor } from "../core/tensor.ts";

export interface AttentionCache {
    keys: Tensor[];
    values: Tensor[];
}

function emptyRows(columns: number): Tensor {
    return new Tensor(new Float32Array(0), [0, columns]);
}

/** Añade filas a una matriz sin construir grafo: el pasado de la caché es constante. */
export function appendRows(previous: Tensor, next: Tensor): Tensor {
    if (previous.rank !== 2 || next.rank !== 2) {
        throw new Error("appendRows espera matrices");
    }
    const columns = previous.shape[1] as number;
    if ((next.shape[1] as number) !== columns) {
        throw new Error("appendRows necesita las mismas columnas");
    }
    const rows = (previous.shape[0] as number) + (next.shape[0] as number);
    const data = new Float32Array(rows * columns);
    data.set(previous.data);
    data.set(next.data, previous.size);
    return new Tensor(data, [rows, columns]);
}

/** Caché por bloque y por cabeza; crece una posición en cada paso. */
export class KVCache {
    readonly layers: AttentionCache[];

    constructor(layers: number, heads: number, headSize: number) {
        if (!Number.isInteger(layers) || layers <= 0) {
            throw new Error(`capas inválidas: ${layers}`);
        }
        if (!Number.isInteger(heads) || heads <= 0) {
            throw new Error(`cabezas inválidas: ${heads}`);
        }
        if (!Number.isInteger(headSize) || headSize <= 0) {
            throw new Error(`tamaño de cabeza inválido: ${headSize}`);
        }
        this.layers = Array.from({ length: layers }, () => ({
            keys: Array.from({ length: heads }, () => emptyRows(headSize)),
            values: Array.from({ length: heads }, () => emptyRows(headSize)),
        }));
    }

    get length(): number {
        return (this.layers[0]?.keys[0]?.shape[0] as number | undefined) ?? 0;
    }
}
