// Teoría: docs/experimentos/familias-tokenizers.md

const UNIT = /\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu;

export interface WordPieceOptions {
    vocabularySize?: number;
    minFrequency?: number;
}

interface Piece {
    text: string;
    continuation: boolean;
    bytes?: Uint8Array;
}

const unitLike = (value: string): boolean => /^[\p{L}\p{N}]+$/u.test(value);

/**
 * WordPiece lossless: vocabulario de caracteres y continuaciones `##`, merges por ganancia de
 * verosimilitud y fallback de bytes para lo no visto.
 */
export class WordPieceTokenizer {
    readonly #pieces: Piece[] = [];
    readonly #index = new Map<string, number>();

    constructor() {}

    get vocabularySize(): number {
        return this.#pieces.length;
    }

    train(text: string, options: WordPieceOptions = {}): void {
        const target = options.vocabularySize ?? 2048;
        const minFrequency = options.minFrequency ?? 2;
        if (!Number.isInteger(target) || target < 256) {
            throw new RangeError(`el vocabulario debe ser al menos 256, no ${target}`);
        }
        if (!Number.isInteger(minFrequency) || minFrequency < 1) {
            throw new RangeError(`la frecuencia mínima debe ser positiva, no ${minFrequency}`);
        }
        this.#pieces.length = 0;
        this.#index.clear();
        const units = text.match(UNIT) ?? [];
        for (const unit of units) {
            if (!unitLike(unit)) {
                this.#add(unit, false);
            } else {
                const characters = [...unit];
                for (let index = 0; index < characters.length; index++) {
                    this.#add(characters[index] as string, index > 0);
                }
            }
        }
        const sequences: number[][] = [];
        for (const unit of units) {
            if (unitLike(unit)) {
                const characters = [...unit];
                sequences.push(
                    characters.map((character, index) =>
                        this.#index.get(`${index > 0 ? "##" : ""}${character}`) as number,
                    ),
                );
            }
        }

        let active = true;
        while (this.#pieces.length < target && active) {
            const counts = new Map<string, number>();
            const leftCounts = new Map<number, number>();
            const rightCounts = new Map<number, number>();
            for (const sequence of sequences) {
                for (let index = 0; index + 1 < sequence.length; index++) {
                    const left = sequence[index] as number;
                    const right = sequence[index + 1] as number;
                    const leftContinuation = this.#pieces[left]?.continuation ?? false;
                    const rightContinuation = this.#pieces[right]?.continuation ?? false;
                    const allowed =
                        !(leftContinuation === false && rightContinuation === false) &&
                        !(leftContinuation && rightContinuation === false);
                    if (allowed) {
                        const key = `${left},${right}`;
                        counts.set(key, (counts.get(key) ?? 0) + 1);
                        leftCounts.set(left, (leftCounts.get(left) ?? 0) + 1);
                        rightCounts.set(right, (rightCounts.get(right) ?? 0) + 1);
                    }
                }
            }
            let bestKey: string | null = null;
            let bestScore = 0;
            for (const [key, count] of counts) {
                if (count >= minFrequency) {
                    const [left, right] = key.split(",").map(Number) as [number, number];
                    const score = count / ((leftCounts.get(left) ?? 1) * (rightCounts.get(right) ?? 1));
                    if (score > bestScore || (score === bestScore && (bestKey === null || key < bestKey))) {
                        bestKey = key;
                        bestScore = score;
                    }
                }
            }
            if (bestKey === null) {
                active = false;
            } else {
                const [left, right] = bestKey.split(",").map(Number) as [number, number];
                const leftPiece = this.#pieces[left] as Piece;
                const rightPiece = this.#pieces[right] as Piece;
                const leftText = leftPiece.continuation ? leftPiece.text.slice(2) : leftPiece.text;
                const mergedText = `${leftText}${rightPiece.text.slice(2)}`;
                const token = this.#add(mergedText, leftPiece.continuation);
                const merged: number[][] = [];
                for (const sequence of sequences) {
                    const next: number[] = [];
                    for (let index = 0; index < sequence.length; ) {
                        if (
                            index + 1 < sequence.length &&
                            sequence[index] === left &&
                            sequence[index + 1] === right
                        ) {
                            next.push(token);
                            index += 2;
                        } else {
                            next.push(sequence[index] as number);
                            index += 1;
                        }
                    }
                    merged.push(next);
                }
                sequences.length = 0;
                sequences.push(...merged);
            }
        }
        for (let byte = 0; byte < 256; byte++) {
            this.#pieces.push({ text: "", continuation: false, bytes: Uint8Array.from([byte]) });
        }
    }

    #add(text: string, continuation: boolean): number {
        const key = `${continuation ? "##" : ""}${text}`;
        const existing = this.#index.get(key);
        if (existing !== undefined) return existing;
        const id = this.#pieces.length;
        this.#pieces.push({ text: key, continuation });
        this.#index.set(key, id);
        return id;
    }

    encode(text: string): number[] {
        const output: number[] = [];
        for (const unit of text.match(UNIT) ?? []) {
            if (!unitLike(unit)) {
                const whole = this.#index.get(unit);
                if (whole !== undefined) {
                    output.push(whole);
                } else {
                    output.push(...this.#fallback(unit));
                }
            } else {
                const characters = [...unit];
                let index = 0;
                let failed = false;
                const unitOutput: number[] = [];
                while (index < characters.length && !failed) {
                    let match: number | null = null;
                    let end = characters.length;
                    while (end > index && match === null) {
                        const substring = characters.slice(index, end).join("");
                        const key = `${index === 0 ? "" : "##"}${substring}`;
                        const token = this.#index.get(key);
                        if (token !== undefined) {
                            match = token;
                            index = end;
                        } else {
                            end -= 1;
                        }
                    }
                    if (match === null) failed = true;
                    else unitOutput.push(match);
                }
                if (failed) output.push(...this.#fallback(unit));
                else output.push(...unitOutput);
            }
        }
        return output;
    }

    #fallback(text: string): number[] {
        const bytes = new TextEncoder().encode(text);
        const start = this.#pieces.length - 256;
        return [...bytes].map((byte) => start + byte);
    }

    decode(tokens: readonly number[]): string {
        const bytes: number[] = [];
        for (const token of tokens) {
            const piece = this.#pieces[token];
            if (!piece) throw new RangeError(`token fuera de vocabulario: ${token}`);
            if (piece.bytes) {
                bytes.push(...piece.bytes);
            } else {
                const text = piece.continuation ? piece.text.slice(2) : piece.text;
                bytes.push(...new TextEncoder().encode(text));
            }
        }
        return new TextDecoder().decode(Uint8Array.from(bytes));
    }
}
