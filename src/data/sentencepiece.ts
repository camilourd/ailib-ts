// Teoría: docs/experimentos/familias-tokenizers.md

import { BPETokenizer } from "./bpe.ts";

const MARKER = "\u2581";
const ESCAPE = "\uE000";
const ESCAPED_MARKER = "M";

export interface SentencePieceOptions {
    vocabularySize?: number;
    minFrequency?: number;
}

/**
 * SentencePiece-BPE lossless: flujo crudo con ▁ para espacios.
 * El marcador literal se escapa como ▁▁ para que decode sea reversible.
 */
export class SentencePieceTokenizer {
    readonly #bpe = new BPETokenizer();

    get vocabularySize(): number {
        return this.#bpe.vocabularySize;
    }

    train(text: string, options: SentencePieceOptions = {}): void {
        this.#bpe.train(this.#normalize(text), {
            vocabularySize: options.vocabularySize ?? 2048,
            minFrequency: options.minFrequency ?? 2,
        });
    }

    #normalize(text: string): string {
        return text
            .replaceAll(ESCAPE, `${ESCAPE}${ESCAPE}`)
            .replaceAll(MARKER, `${ESCAPE}${ESCAPED_MARKER}`)
            .replaceAll(" ", MARKER);
    }

    #restore(text: string): string {
        let output = "";
        for (let index = 0; index < text.length; index++) {
            const character = text[index] as string;
            if (character === ESCAPE) {
                const next = text[index + 1];
                if (next === ESCAPE) {
                    output += ESCAPE;
                    index += 1;
                } else if (next === ESCAPED_MARKER) {
                    output += MARKER;
                    index += 1;
                } else {
                    output += ESCAPE;
                }
            } else if (character === MARKER) {
                output += " ";
            } else {
                output += character;
            }
        }
        return output;
    }

    encode(text: string): number[] {
        return this.#bpe.encode(this.#normalize(text));
    }

    decode(tokens: readonly number[]): string {
        return this.#restore(this.#bpe.decode(tokens));
    }
}
