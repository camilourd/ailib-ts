// F11.6a: barrido de vocabulario 2k/8k/16k con el train completo.
// Sin LM: exactitud, compresión, frontera y ED.

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

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, Math.floor(quijote.length * 0.7));
const test = quijote.slice(Math.floor(quijote.length * 0.85));
const reference = JSON.parse(readFileSync("data/morphology-reference-reviewed.json", "utf8")) as {
    records: Reference[];
};

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

function measure(tokenizer: TokenizerLike): {
    tokensPerChar: number;
    boundaryF1: number;
    editDistance: number;
    exact: boolean;
} {
    const testTokens = tokenizer.encode(test);
    let truePositives = 0;
    let predictedTotal = 0;
    let support = 0;
    let distance = 0;
    let distanceSupport = 0;
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
            distanceSupport += 1;
        }
    }
    const precision = predictedTotal === 0 ? 0 : truePositives / predictedTotal;
    const recall = support === 0 ? 0 : truePositives / support;
    return {
        tokensPerChar: testTokens.length / [...test].length,
        boundaryF1: precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall),
        editDistance: distanceSupport === 0 ? 0 : distance / distanceSupport,
        exact:
            tokenizer.decode(tokenizer.encode(train)) === train &&
            tokenizer.decode(testTokens) === test &&
            tokenizer.decode(tokenizer.encode(external)) === external,
    };
}

function bpe(vocabularySize: number): TokenizerLike {
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

const rows = [];
for (const vocabularySize of [2048, 8192, 16384]) {
    const variants: Array<[string, () => TokenizerLike]> = [
        [`bpe-${vocabularySize}`, () => bpe(vocabularySize)],
        [`sentencepiece-${vocabularySize}`, () => {
            const tokenizer = new SentencePieceTokenizer();
            tokenizer.train(train, { vocabularySize });
            return tokenizer;
        }],
        [`wordpiece-${vocabularySize}`, () => {
            const tokenizer = new WordPieceTokenizer();
            tokenizer.train(train, { vocabularySize, minFrequency: 5 });
            return tokenizer;
        }],
        [`unigram-${vocabularySize}`, () => {
            const tokenizer = new UnigramTokenizer();
            tokenizer.train(train, { vocabularySize, iterations: 2, minFrequency: 2 });
            return tokenizer;
        }],
    ];
    for (const [name, build] of variants) {
        const started = Date.now();
        const tokenizer = build();
        const metrics = measure(tokenizer);
        rows.push({
            name,
            requested: vocabularySize,
            actual: tokenizer.vocabularySize,
            ...metrics,
            seconds: (Date.now() - started) / 1000,
        });
        console.log(
            `${name.padEnd(24)} real ${String(tokenizer.vocabularySize).padStart(5)} · ` +
                `tokens/car ${metrics.tokensPerChar.toFixed(3)} · F1 ${metrics.boundaryF1.toFixed(3)} · ` +
                `ED ${metrics.editDistance.toFixed(3)} · ${((Date.now() - started) / 1000).toFixed(1)}s`,
        );
        writeFileSync("data/tokenizer-vocab-sweep.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
    }
}
console.log("guardado: data/tokenizer-vocab-sweep.json");
