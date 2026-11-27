// Teoría: docs/teoria/nucleo/bpe.md

import { BPETokenizer } from "./bpe.ts";

const SPLIT =
    /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** BPE con pre-tokenización estilo GPT-2: los merges no cruzan palabras/números/símbolos. */
export class PreTokenizedBPETokenizer {
    readonly #bpe = new BPETokenizer();

    get vocabularySize(): number {
        return this.#bpe.vocabularySize;
    }

    get merges(): readonly { left: number; right: number; token: number }[] {
        return this.#bpe.merges;
    }

    train(text: string, options: { vocabularySize?: number; minFrequency?: number } = {}): void {
        this.#bpe.train(chunks(text).join("\u0000"), options);
    }

    encode(text: string): number[] {
        return chunks(text).flatMap((chunk) => this.#bpe.encode(chunk));
    }

    decode(tokens: readonly number[]): string {
        return this.#bpe.decode(tokens);
    }
}

function chunks(text: string): string[] {
    const result = text.match(SPLIT);
    return result ?? [];
}
