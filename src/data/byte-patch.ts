// Teoría: docs/experimentos/familias-tokenizers.md

import { ByteTokenizer } from "./tokenizer.ts";

export interface EntropyPatchOptions {
    threshold?: number;
    maxPatch?: number;
    vocabularySize?: number;
}

const keyOf = (bytes: Uint8Array): string => [...bytes].join(",");

/**
 * Patcher BLT-style: agrupa bytes mientras la entropía del siguiente byte es baja.
 * No es BLT completo: no aprende un modelo de parches, usa un n-gram de bytes.
 */
export class EntropyByteTokenizer {
    readonly #bytes = new ByteTokenizer();
    readonly #counts = new Map<string, Map<number, number>>();
    readonly #patches: Uint8Array[] = [];
    readonly #patchIds = new Map<string, number>();
    #threshold = 2;
    #maxPatch = 8;

    get vocabularySize(): number {
        return 256 + this.#patches.length;
    }

    train(text: string, options: EntropyPatchOptions = {}): void {
        this.#threshold = options.threshold ?? 2;
        this.#maxPatch = options.maxPatch ?? 8;
        const vocabularySize = options.vocabularySize ?? 2048;
        this.#counts.clear();
        this.#patches.length = 0;
        this.#patchIds.clear();
        const bytes = Uint8Array.from(this.#bytes.encode(text));
        for (let position = 0; position < bytes.length; position++) {
            const context = this.#context(bytes, position);
            let distribution = this.#counts.get(context);
            if (!distribution) {
                distribution = new Map();
                this.#counts.set(context, distribution);
            }
            distribution.set(bytes[position] as number, (distribution.get(bytes[position] as number) ?? 0) + 1);
        }
        const patchCounts = new Map<string, { bytes: Uint8Array; count: number }>();
        for (const patch of this.#patch(bytes)) {
            const key = keyOf(patch);
            const current = patchCounts.get(key);
            patchCounts.set(key, { bytes: patch, count: (current?.count ?? 0) + 1 });
        }
        const chosen = [...patchCounts.entries()]
            .filter(([key]) => key.includes(","))
            .sort((first, second) => second[1].count - first[1].count || first[0].localeCompare(second[0]))
            .slice(0, Math.max(0, vocabularySize - 256));
        for (const [key, value] of chosen) {
            this.#patchIds.set(key, 256 + this.#patches.length);
            this.#patches.push(value.bytes);
        }
    }

    #context(bytes: Uint8Array, position: number): string {
        if (position === 0) return "^,^";
        if (position === 1) return `^,${bytes[0]}`;
        return `${bytes[position - 2]},${bytes[position - 1]}`;
    }

    #entropy(context: string): number {
        const distribution = this.#counts.get(context);
        if (!distribution) return 8;
        let total = 0;
        for (const count of distribution.values()) total += count;
        let entropy = 0;
        for (const count of distribution.values()) {
            const probability = count / total;
            entropy -= probability * Math.log2(probability);
        }
        return entropy;
    }

    #patch(bytes: Uint8Array): Uint8Array[] {
        const patches: Uint8Array[] = [];
        let start = 0;
        while (start < bytes.length) {
            let end = start + 1;
            while (
                end < bytes.length &&
                end - start < this.#maxPatch &&
                this.#entropy(this.#context(bytes, end)) < this.#threshold
            ) {
                end += 1;
            }
            patches.push(Uint8Array.from(bytes.slice(start, end)));
            start = end;
        }
        return patches;
    }

    encode(text: string): number[] {
        const output: number[] = [];
        for (const patch of this.#patch(Uint8Array.from(this.#bytes.encode(text)))) {
            const key = keyOf(patch);
            const id = this.#patchIds.get(key);
            if (id !== undefined) {
                output.push(id);
            } else {
                output.push(...patch);
            }
        }
        return output;
    }

    decode(tokens: readonly number[]): string {
        const bytes: number[] = [];
        for (const token of tokens) {
            if (token < 256) {
                bytes.push(token);
            } else {
                const patch = this.#patches[token - 256];
                if (!patch) throw new RangeError(`token fuera de vocabulario: ${token}`);
                bytes.push(...patch);
            }
        }
        return this.#bytes.decode(bytes);
    }
}
