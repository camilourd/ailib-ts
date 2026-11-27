// F12.1: zoo de tokenizers por dominio y vocabulario real.
// Sin LM: exactitud, compresión y frontera morfológica en narrativo.

import { readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { SentencePieceTokenizer } from "../../src/data/sentencepiece.ts";
import { UnigramTokenizer } from "../../src/data/unigram.ts";
import { WordPieceTokenizer } from "../../src/data/wordpiece.ts";

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

interface Reference {
    form: string;
    stem: string;
    suffix: string;
    confidence: string;
}

const domains = ["narrativo", "tecnico", "codigo"] as const;
const vocabularies = [2048, 8192, 16384];
const reference = JSON.parse(readFileSync("data/morphology-reference-reviewed.json", "utf8")) as {
    records: Reference[];
};

function bpe(train: string, vocabularySize: number): TokenizerLike {
    const bpe = new BPETokenizer();
    bpe.train(train, { vocabularySize });
    const tokens: Uint8Array[] = [];
    const seen = new Set<string>();
    for (let byte = 0; byte < 256; byte++) {
        tokens.push(Uint8Array.from([byte]));
        seen.add(String.fromCharCode(byte));
    }
    for (let id = 0; id < bpe.vocabularySize; id++) {
        const text = bpe.decode([id]);
        if (!text.includes("\uFFFD")) {
            const bytes = new TextEncoder().encode(text);
            const key = String.fromCharCode(...bytes);
            if (!seen.has(key)) {
                seen.add(key);
                tokens.push(bytes);
            }
        }
    }
    const tokenizer = new LosslessByteTokenizer(tokens.length);
    for (const token of tokens) tokenizer.add(token);
    return tokenizer;
}

function editDistance(first: string[], second: string[]): number {
    const previous = Array.from({ length: second.length + 1 }, (_, index) => index);
    for (let row = 1; row <= first.length; row++) {
        let diagonal = previous[0] as number;
        previous[0] = row;
        for (let column = 1; column <= second.length; column++) {
            const above = previous[column] as number;
            previous[column] = Math.min(
                (previous[column] as number) + 1,
                (previous[column - 1] as number) + 1,
                diagonal + (first[row - 1] === second[column - 1] ? 0 : 1),
            );
            diagonal = above;
        }
    }
    return previous[second.length] as number;
}

function morphology(tokenizer: TokenizerLike): { f1: number; ed: number } {
    let truePositives = 0;
    let predictedTotal = 0;
    let support = 0;
    let distance = 0;
    let count = 0;
    for (const record of reference.records) {
        if (record.confidence === "alta" && record.stem && record.suffix) {
            const pieces = tokenizer.encode(record.form).map((token) => tokenizer.decode([token]));
            const boundaries = new Set<number>();
            let offset = 0;
            for (let index = 0; index < pieces.length - 1; index++) {
                offset += (pieces[index] as string).length;
                boundaries.add(offset);
            }
            predictedTotal += boundaries.size;
            support += 1;
            if (boundaries.has(record.stem.length)) truePositives += 1;
            distance += editDistance([record.stem, record.suffix], pieces) / Math.max(2, pieces.length);
            count += 1;
        }
    }
    const precision = predictedTotal === 0 ? 0 : truePositives / predictedTotal;
    const recall = support === 0 ? 0 : truePositives / support;
    return {
        f1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
        ed: count === 0 ? 0 : distance / count,
    };
}

const rows: Array<Record<string, number | string | boolean>> = [];
for (const domain of domains) {
    const train = readFileSync(`data/corpora/${domain}-train.txt`, "utf8");
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    for (const vocabularySize of vocabularies) {
        const variants: Array<[string, () => TokenizerLike]> = [
            ["bpe", () => bpe(train, vocabularySize)],
            ["sentencepiece", () => {
                const tokenizer = new SentencePieceTokenizer();
                tokenizer.train(train, { vocabularySize });
                return tokenizer;
            }],
            ["wordpiece", () => {
                const tokenizer = new WordPieceTokenizer();
                tokenizer.train(train, { vocabularySize, minFrequency: 5 });
                return tokenizer;
            }],
            ["unigram", () => {
                const tokenizer = new UnigramTokenizer();
                tokenizer.train(train, { vocabularySize, iterations: 2, minFrequency: 2 });
                return tokenizer;
            }],
        ];
        for (const [family, build] of variants) {
            const started = Date.now();
            const tokenizer = build();
            const testTokens = tokenizer.encode(test);
            const exact =
                tokenizer.decode(tokenizer.encode(train)) === train &&
                tokenizer.decode(testTokens) === test;
            const morph = domain === "narrativo" ? morphology(tokenizer) : { f1: 0, ed: 0 };
            rows.push({
                domain,
                family,
                requested: vocabularySize,
                actual: tokenizer.vocabularySize,
                tokensPerChar: testTokens.length / [...test].length,
                boundaryF1: morph.f1,
                editDistance: morph.ed,
                exact,
                seconds: (Date.now() - started) / 1000,
            });
            console.log(
                `${domain.padEnd(9)} ${family.padEnd(14)} objetivo ${String(vocabularySize).padStart(5)} · ` +
                    `real ${String(tokenizer.vocabularySize).padStart(5)} · tokens/car ${(testTokens.length / [...test].length).toFixed(3)} · exacto ${exact}`,
            );
            writeFileSync("data/tokenizer-zoo.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
        }
    }
}
console.log("guardado: data/tokenizer-zoo.json");
