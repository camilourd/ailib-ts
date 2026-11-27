// Teoría: docs/teoria/nucleo/bpe.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { BPETokenizer } from "./bpe.ts";

export interface BPEDropoutOptions {
    rate?: number;
    seed?: number;
    rng?: RandomNumberGenerator;
}

/**
 * BPE con dropout de merges: en entrenamiento cada merge se aplica con probabilidad 1-rate.
 * `decode` sigue siendo lossless; no se usa dropout al evaluar.
 */
export class BPEDropoutTokenizer {
    readonly bpe = new BPETokenizer();
    readonly #rate: number;
    readonly #rng: RandomNumberGenerator;

    constructor(options: BPEDropoutOptions = {}) {
        const rate = options.rate ?? 0.1;
        if (!Number.isFinite(rate) || rate < 0 || rate >= 1) {
            throw new RangeError(`dropout debe estar en [0, 1), no ${rate}`);
        }
        this.#rate = rate;
        this.#rng = options.rng ?? createRng(options.seed ?? 0);
    }

    get vocabularySize(): number {
        return this.bpe.vocabularySize;
    }

    train(text: string, options: { vocabularySize?: number; minFrequency?: number } = {}): void {
        this.bpe.train(text, options);
    }

    encode(text: string, options: { dropout?: boolean } = {}): number[] {
        let sequence = Array.from(this.bpe.base.encode(text));
        for (const merge of this.bpe.merges) {
            const skip = (options.dropout ?? false) && this.#rng.next() < this.#rate;
            if (!skip) {
                const merged: number[] = [];
                for (let index = 0; index < sequence.length; ) {
                    if (
                        index + 1 < sequence.length &&
                        sequence[index] === merge.left &&
                        sequence[index + 1] === merge.right
                    ) {
                        merged.push(merge.token);
                        index += 2;
                    } else {
                        merged.push(sequence[index] as number);
                        index += 1;
                    }
                }
                sequence = merged;
            }
        }
        return sequence;
    }

    decode(tokens: readonly number[]): string {
        return this.bpe.decode(tokens);
    }
}
