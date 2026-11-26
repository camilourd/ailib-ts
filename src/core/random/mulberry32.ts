// Teoría: docs/teoria/nucleo/rng.md

import { Uint32RandomGenerator } from "./rng.ts";

/** mulberry32: estado de 32 bits y periodo 2^32. */
export class Mulberry32 extends Uint32RandomGenerator {
    #state: number;

    constructor(seed: number) {
        super();
        this.#state = seed >>> 0;
    }

    protected nextUint32(): number {
        this.#state = (this.#state + 0x6d2b79f5) >>> 0;
        let t = Math.imul(this.#state ^ (this.#state >>> 15), this.#state | 1);
        t = (t ^ (t + Math.imul(t ^ (t >>> 7), t | 61))) >>> 0;
        return (t ^ (t >>> 14)) >>> 0;
    }
}
