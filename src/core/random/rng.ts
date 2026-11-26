// Teoría: docs/teoria/nucleo/rng.md

/** Contrato que usa el resto de la librería: no depende del algoritmo. */
export interface RandomNumberGenerator {
    /** Siguiente número uniforme en [0, 1). */
    next(): number;
    /** Entero uniforme en [0, maxExclusive). */
    int(maxExclusive: number): number;
    /** Copia mezclada de `values` (Fisher-Yates); no muta la entrada. */
    shuffle<T>(values: readonly T[]): T[];
}

const TWO_POW_32 = 0x100000000;

/**
 * Base de los generadores de 32 bits (Template Method): `next`, `int` y `shuffle` se implementan
 * una sola vez aquí; cada algoritmo solo aporta su paso crudo en `nextUint32`.
 */
export abstract class Uint32RandomGenerator implements RandomNumberGenerator {
    protected abstract nextUint32(): number;

    next(): number {
        return this.nextUint32() / TWO_POW_32;
    }

    int(maxExclusive: number): number {
        if (!Number.isInteger(maxExclusive) || maxExclusive <= 0 || maxExclusive > TWO_POW_32) {
            throw new RangeError(`maxExclusive debe ser un entero en (0, 2^32], no ${maxExclusive}`);
        }
        const limit = TWO_POW_32 - (TWO_POW_32 % maxExclusive);
        let value = this.nextUint32();
        while (value >= limit) value = this.nextUint32();
        return value % maxExclusive;
    }

    shuffle<T>(values: readonly T[]): T[] {
        const copy = [...values];
        // i y j están siempre dentro del rango por construcción; el cast solo satisface
        // noUncheckedIndexedAccess.
        for (let i = copy.length - 1; i > 0; i--) {
            const j = this.int(i + 1);
            const swap = copy[i] as T;
            copy[i] = copy[j] as T;
            copy[j] = swap;
        }
        return copy;
    }
}
