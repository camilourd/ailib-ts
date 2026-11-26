// Teoría: docs/teoria/nucleo/bpe.md

import { ByteTokenizer } from "./tokenizer.ts";

export interface BPEOptions {
    vocabularySize?: number;
    minFrequency?: number;
}

export interface Merge {
    left: number;
    right: number;
    token: number;
}

/** Byte Pair Encoding: parte de 256 bytes y fusiona pares frecuentes. */
export class BPETokenizer {
    readonly base = new ByteTokenizer();
    readonly #merges: Merge[] = [];
    #tokenBytes: Uint8Array[];
    #vocabularySize = 256;

    constructor() {
        this.#tokenBytes = Array.from({ length: 256 }, (_, byte) => new Uint8Array([byte]));
    }

    get vocabularySize(): number {
        return this.#vocabularySize;
    }

    get merges(): readonly Merge[] {
        return this.#merges;
    }

    /** Aprende merges sobre `text` hasta el vocabulario pedido. */
    train(text: string, options: BPEOptions = {}): void {
        const target = options.vocabularySize ?? 512;
        const minFrequency = options.minFrequency ?? 2;
        if (!Number.isInteger(target) || target < 256) {
            throw new RangeError(`el vocabulario debe ser al menos 256, no ${target}`);
        }
        if (!Number.isInteger(minFrequency) || minFrequency < 1) {
            throw new RangeError(`la frecuencia mínima debe ser positiva, no ${minFrequency}`);
        }
        this.#merges.length = 0;
        this.#tokenBytes = Array.from({ length: 256 }, (_, byte) => new Uint8Array([byte]));
        this.#vocabularySize = 256;

        let sequence = Array.from(this.base.encode(text));
        let training = true;
        while (training && this.#vocabularySize < target && sequence.length >= 2) {
            const counts = new Map<string, number>();
            for (let index = 0; index + 1 < sequence.length; index++) {
                const key = `${sequence[index]},${sequence[index + 1]}`;
                counts.set(key, (counts.get(key) ?? 0) + 1);
            }
            let bestKey: string | null = null;
            let bestCount = 0;
            for (const [key, count] of counts) {
                if (
                    count > bestCount ||
                    (count === bestCount && (bestKey === null || key < bestKey))
                ) {
                    bestKey = key;
                    bestCount = count;
                }
            }
            if (bestKey === null || bestCount < minFrequency) {
                training = false;
            } else {
                const [left, right] = bestKey.split(",").map(Number) as [number, number];
                const token = this.#vocabularySize;
                const first = this.#tokenBytes[left] as Uint8Array;
                const second = this.#tokenBytes[right] as Uint8Array;
                const bytes = new Uint8Array(first.length + second.length);
                bytes.set(first);
                bytes.set(second, first.length);
                this.#merges.push({ left, right, token });
                this.#tokenBytes.push(bytes);
                this.#vocabularySize += 1;

                const merged: number[] = [];
                for (let index = 0; index < sequence.length; ) {
                    if (
                        index + 1 < sequence.length &&
                        sequence[index] === left &&
                        sequence[index + 1] === right
                    ) {
                        merged.push(token);
                        index += 2;
                    } else {
                        merged.push(sequence[index] as number);
                        index += 1;
                    }
                }
                sequence = merged;
            }
        }
    }

    encode(text: string): number[] {
        let sequence = Array.from(this.base.encode(text));
        for (const merge of this.#merges) {
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
        return sequence;
    }

    decode(tokens: readonly number[]): string {
        const bytes: number[] = [];
        for (const token of tokens) {
            if (!Number.isInteger(token) || token < 0 || token >= this.#vocabularySize) {
                throw new RangeError(`token inválido ${token}; fuera del vocabulario`);
            }
            const chunk = this.#tokenBytes[token] as Uint8Array;
            for (const byte of chunk) bytes.push(byte);
        }
        return this.base.decode(bytes);
    }
}
