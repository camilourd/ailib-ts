// Teoría: docs/experimentos/familias-tokenizers.md

const UNIT = /\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu;
const unitLike = (value: string): boolean => /^[\p{L}\p{N}]+$/u.test(value);

export interface UnigramOptions {
    vocabularySize?: number;
    maxTokenLength?: number;
    iterations?: number;
    minFrequency?: number;
}

/**
 * Unigram lossless: candidatos por substrings, EM de probabilidades y Viterbi.
 * Los bytes no vistos caen a un fallback de 256 bytes.
 */
export class UnigramTokenizer {
    #tokens: string[] = [];
    #probs: number[] = [];
    #index = new Map<string, number>();
    #byteStart = 0;
    #maxTokenLength = 16;

    get vocabularySize(): number {
        return this.#tokens.length + (this.#byteStart === 0 ? 0 : 256);
    }

    train(text: string, options: UnigramOptions = {}): void {
        const target = options.vocabularySize ?? 2048;
        const maxTokenLength = options.maxTokenLength ?? 16;
        const iterations = options.iterations ?? 2;
        const minFrequency = options.minFrequency ?? 2;
        if (!Number.isInteger(target) || target < 256) {
            throw new RangeError(`el vocabulario debe ser al menos 256, no ${target}`);
        }
        if (!Number.isInteger(maxTokenLength) || maxTokenLength < 1) {
            throw new RangeError(`maxTokenLength inválido: ${maxTokenLength}`);
        }
        this.#maxTokenLength = maxTokenLength;
        this.#tokens = [];
        this.#probs = [];
        this.#index.clear();
        this.#byteStart = 0;

        const units = text.match(UNIT) ?? [];
        const counts = new Map<string, number>();
        for (const unit of units) {
            if (!unitLike(unit)) {
                counts.set(unit, (counts.get(unit) ?? 0) + 1);
            } else {
                const characters = [...unit];
                for (let start = 0; start < characters.length; start++) {
                    let candidate = "";
                    for (
                        let end = start;
                        end < characters.length && end - start < maxTokenLength;
                        end++
                    ) {
                        candidate += characters[end] as string;
                        counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
                    }
                }
            }
        }
        const chars = [...new Set([...counts.keys()].filter((value) => [...value].length === 1))];
        const candidates = [...counts.entries()]
            .filter(([value, count]) => count >= minFrequency && [...value].length > 1)
            .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
            .slice(0, Math.max(0, target - chars.length))
            .map(([value]) => value);
        for (const value of [...chars, ...candidates]) this.#add(value, counts.get(value) ?? 1);

        for (let iteration = 0; iteration < iterations; iteration++) {
            const selected = new Array<number>(this.#tokens.length).fill(0.5);
            for (const unit of units) {
                if (unitLike(unit)) {
                    for (const token of this.#viterbi(unit)) selected[token] = (selected[token] as number) + 1;
                } else {
                    const id = this.#index.get(unit);
                    if (id !== undefined) selected[id] = (selected[id] as number) + 1;
                }
            }
            const total = selected.reduce((sum, value) => sum + value, 0);
            this.#probs = selected.map((value) => value / total);
        }

        this.#byteStart = this.#tokens.length;
    }

    #add(value: string, count: number): void {
        const id = this.#tokens.length;
        this.#tokens.push(value);
        this.#probs.push(count);
        this.#index.set(value, id);
    }

    #viterbi(unit: string): number[] {
        const characters = [...unit];
        const size = characters.length;
        const costs = new Array<number>(size + 1).fill(Number.POSITIVE_INFINITY);
        const previous = new Array<number>(size + 1).fill(-1);
        const chosen = new Array<number>(size + 1).fill(-1);
        costs[0] = 0;
        for (let start = 0; start < size; start++) {
            const base = costs[start] as number;
            if (Number.isFinite(base)) {
                let candidate = "";
                for (
                    let end = start;
                    end < size && end - start < this.#maxTokenLength;
                    end++
                ) {
                    candidate += characters[end] as string;
                    const id = this.#index.get(candidate);
                    if (id !== undefined) {
                        const price = base - Math.log(Math.max(this.#probs[id] as number, 1e-9));
                        if (price < (costs[end + 1] as number)) {
                            costs[end + 1] = price;
                            previous[end + 1] = start;
                            chosen[end + 1] = id;
                        }
                    }
                }
            }
        }
        if (!Number.isFinite(costs[size] as number)) return [];
        const tokens: number[] = [];
        let cursor = size;
        while (cursor > 0) {
            tokens.push(chosen[cursor] as number);
            cursor = previous[cursor] as number;
        }
        return tokens.reverse();
    }

    encode(text: string): number[] {
        const output: number[] = [];
        for (const unit of text.match(UNIT) ?? []) {
            if (unitLike(unit)) {
                const tokens = this.#viterbi(unit);
                if (tokens.length > 0) {
                    output.push(...tokens);
                } else {
                    output.push(...this.#fallback(unit));
                }
            } else {
                const id = this.#index.get(unit);
                if (id !== undefined) output.push(id);
                else output.push(...this.#fallback(unit));
            }
        }
        return output;
    }

    #fallback(text: string): number[] {
        return [...new TextEncoder().encode(text)].map((byte) => this.#byteStart + byte);
    }

    decode(tokens: readonly number[]): string {
        const bytes: number[] = [];
        for (const token of tokens) {
            if (token < this.#byteStart) {
                bytes.push(...new TextEncoder().encode(this.#tokens[token] as string));
            } else if (token < this.#byteStart + 256) {
                bytes.push(token - this.#byteStart);
            } else {
                throw new RangeError(`token fuera de vocabulario: ${token}`);
            }
        }
        return new TextDecoder().decode(Uint8Array.from(bytes));
    }
}
