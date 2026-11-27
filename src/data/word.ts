// Teoría: docs/teoria/nucleo/tokenizadores.md

const UNIT = /\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu;

export interface WordTokenizerOptions {
    minFrequency?: number;
}

/** Tokenizer por palabras, números, signos y espacios; lo no visto cae en `<unk>`. */
export class WordTokenizer {
    #tokens: string[] = ["<unk>"];
    #index = new Map<string, number>([["<unk>", 0]]);

    get vocabularySize(): number {
        return this.#tokens.length;
    }

    train(text: string, options: WordTokenizerOptions = {}): this {
        if (text.length === 0) throw new Error("no se puede entrenar con texto vacío");
        const minFrequency = options.minFrequency ?? 2;
        if (!Number.isInteger(minFrequency) || minFrequency <= 0) {
            throw new Error(`frecuencia mínima inválida: ${minFrequency}`);
        }
        const counts = new Map<string, number>();
        for (const unit of text.match(UNIT) ?? []) {
            counts.set(unit, (counts.get(unit) ?? 0) + 1);
        }
        const ordered = [...counts.entries()]
            .filter(([, count]) => count >= minFrequency)
            .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
            .map(([unit]) => unit);
        this.#tokens = ["<unk>", ...ordered];
        this.#index = new Map(this.#tokens.map((token, index) => [token, index]));
        return this;
    }

    encode(text: string): number[] {
        return (text.match(UNIT) ?? []).map((unit) => this.#index.get(unit) ?? 0);
    }

    decode(tokens: readonly number[]): string {
        let text = "";
        for (const token of tokens) {
            if (!Number.isInteger(token) || token < 0 || token >= this.#tokens.length) {
                throw new RangeError(`token fuera de vocabulario: ${token}`);
            }
            text += token === 0 ? "\uFFFD" : (this.#tokens[token] as string);
        }
        return text;
    }
}
