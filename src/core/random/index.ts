// Teoría: docs/teoria/nucleo/rng.md

import { Mulberry32 } from "./mulberry32.ts";
import type { RandomNumberGenerator } from "./rng.ts";

export { Mulberry32 } from "./mulberry32.ts";
export { Uint32RandomGenerator } from "./rng.ts";
export type { RandomNumberGenerator } from "./rng.ts";

/** Generador por defecto: la misma semilla produce siempre la misma secuencia. */
export function createRng(seed: number): RandomNumberGenerator {
    return new Mulberry32(seed);
}
