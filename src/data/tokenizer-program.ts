// Teoría: docs/experimentos/evolucion-tokenizers.md

import { BPETokenizer } from "./bpe.ts";
import { ByteTokenizer } from "./tokenizer.ts";
import { LosslessByteTokenizer } from "./lossless.ts";

export type TokenizerFamily = "byte" | "char" | "word" | "dict" | "bpe";

export interface TokenizerProgram {
    family: TokenizerFamily;
    vocabularySize: number;
    maxTokenBytes: number;
    minFrequency: number;
}

export interface ProgramTokenizer {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const FAMILIES: TokenizerFamily[] = ["byte", "char", "word", "dict", "bpe"];
const VOCABULARIES = [512, 1024, 2048, 4096, 8192, 16384, 32768];
const LENGTHS = [32, 64, 128];
const FREQUENCIES = [1, 2, 5, 10];

/** Genoma mínimo de F9.1; el mapeo GE/SGE/πGE llega en F9.2. */
export function programFromGenome(genome: readonly number[]): TokenizerProgram {
    if (genome.length < 4) throw new Error("el genoma necesita al menos 4 codones");
    return {
        family: FAMILIES[(genome[0] as number) % FAMILIES.length] as TokenizerFamily,
        vocabularySize: VOCABULARIES[(genome[1] as number) % VOCABULARIES.length] as number,
        maxTokenBytes: LENGTHS[(genome[2] as number) % LENGTHS.length] as number,
        minFrequency: FREQUENCIES[(genome[3] as number) % FREQUENCIES.length] as number,
    };
}

function units(text: string): string[] {
    return text.match(/\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu) ?? [];
}

function byteLength(value: string): number {
    return new TextEncoder().encode(value).length;
}

function addByFrequency(
    tokenizer: LosslessByteTokenizer,
    counts: Map<string, number>,
    minFrequency: number,
    maxTokenBytes: number,
): void {
    const ordered = [...counts.entries()]
        .filter(([value, count]) => count >= minFrequency && byteLength(value) <= maxTokenBytes)
        .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]));
    for (const [value] of ordered) {
        if (tokenizer.add(new TextEncoder().encode(value)) === null) return;
    }
}

/** Construye el tokenizer que describe el programa. */
export function buildTokenizer(program: TokenizerProgram, train: string): ProgramTokenizer {
    if (program.family === "byte") return new ByteTokenizer();
    if (program.family === "bpe") {
        const bpe = new BPETokenizer();
        bpe.train(train, {
            vocabularySize: Math.max(256, program.vocabularySize),
            minFrequency: program.minFrequency,
        });
        return bpe;
    }

    const tokenizer = new LosslessByteTokenizer(program.vocabularySize);
    if (program.family === "char") {
        const counts = new Map<string, number>();
        for (const character of train) {
            counts.set(character, (counts.get(character) ?? 0) + 1);
        }
        addByFrequency(tokenizer, counts, 1, program.maxTokenBytes);
        return tokenizer;
    }

    if (program.family === "word") {
        const counts = new Map<string, number>();
        for (const unit of units(train)) {
            counts.set(unit, (counts.get(unit) ?? 0) + 1);
        }
        addByFrequency(tokenizer, counts, program.minFrequency, program.maxTokenBytes);
        return tokenizer;
    }

    const counts = new Map<string, number>();
    for (const unit of units(train)) {
        const characters = [...unit];
        for (let start = 0; start < characters.length; start++) {
            let candidate = "";
            for (
                let end = start;
                end < characters.length &&
                byteLength(candidate + characters[end]) <= program.maxTokenBytes;
                end++
            ) {
                candidate += characters[end] as string;
                counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
            }
        }
    }
    addByFrequency(tokenizer, counts, program.minFrequency, program.maxTokenBytes);
    return tokenizer;
}

export function isLossless(tokenizer: ProgramTokenizer, texts: readonly string[]): boolean {
    return texts.every((text) => tokenizer.decode(tokenizer.encode(text)) === text);
}
