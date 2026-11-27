// Teoría: docs/teoria/nucleo/tokenizadores.md

/** Tokenizer por caracteres Unicode (code points), con `<unk>` para lo no visto. */
export class CharTokenizer {
    #characters: string[] = ["<unk>"];
    #index = new Map<string, number>([["<unk>", 0]]);

    get vocabularySize(): number {
        return this.#characters.length;
    }

    train(text: string): this {
        if (text.length === 0) throw new Error("no se puede entrenar con texto vacío");
        const counts = new Map<string, number>();
        for (const character of text) {
            counts.set(character, (counts.get(character) ?? 0) + 1);
        }
        const ordered = [...counts.entries()]
            .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
            .map(([character]) => character);
        this.#characters = ["<unk>", ...ordered];
        this.#index = new Map(this.#characters.map((character, index) => [character, index]));
        return this;
    }

    encode(text: string): number[] {
        const tokens: number[] = [];
        for (const character of text) {
            tokens.push(this.#index.get(character) ?? 0);
        }
        return tokens;
    }

    decode(tokens: readonly number[]): string {
        let text = "";
        for (const token of tokens) {
            if (!Number.isInteger(token) || token < 0 || token >= this.#characters.length) {
                throw new RangeError(`token fuera de vocabulario: ${token}`);
            }
            text += token === 0 ? "\uFFFD" : (this.#characters[token] as string);
        }
        return text;
    }
}
