// Teoría: docs/teoria/nucleo/tokenizer.md

export interface Tokenizer {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Tokenizer byte: cada token es un byte UTF-8 (0..255). */
export class ByteTokenizer implements Tokenizer {
    readonly vocabularySize = 256;

    encode(text: string): number[] {
        return Array.from(encoder.encode(text));
    }

    decode(tokens: readonly number[]): string {
        const bytes = new Uint8Array(tokens.length);
        tokens.forEach((token, index) => {
            if (!Number.isInteger(token) || token < 0 || token >= 256) {
                throw new RangeError(`token inválido ${token} en ${index}; debe ser un byte 0..255`);
            }
            bytes[index] = token;
        });
        return decoder.decode(bytes);
    }
}
