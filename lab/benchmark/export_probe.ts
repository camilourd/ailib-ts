// F10.2: exporta las frases de la referencia con sus tokens y la posición del verbo.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { buildTokenizer, programFromGenome } from "../../src/data/tokenizer-program.ts";

interface Reference {
    form: string;
    sentence: string;
    tense: string;
    source: string;
}

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const testStart = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const test = quijote.slice(testStart);
const reference = JSON.parse(readFileSync("data/morphology-reference-reviewed.json", "utf8")) as {
    records: Reference[];
};

const endings = [
    "ábamos", "ábais", "abais", "aban", "abas", "aba",
    "íamos", "íais", "ían", "ías", "ía",
    "ieron", "ió", "aste", "aron", "ado", "ido", "ando", "iendo",
    "aremos", "aréis", "arán", "ará", "eremos", "eréis", "erán", "erá",
    "iremos", "iréis", "irán", "irá", "aría", "arías", "arían",
    "amos", "áis", "an", "as", "en", "es", "emos", "éis",
];

const bpe = new BPETokenizer();
bpe.train(train, { vocabularySize: 2048, minFrequency: 2 });
const baseTokens: Uint8Array[] = [];
const seen = new Set<string>();
for (let byte = 0; byte < 256; byte++) {
    baseTokens.push(Uint8Array.from([byte]));
    seen.add(String.fromCharCode(byte));
}
for (let id = 0; id < bpe.vocabularySize; id++) {
    const text = bpe.decode([id]);
    if (!text.includes("\uFFFD")) {
        const bytes = new TextEncoder().encode(text);
        const key = String.fromCharCode(...bytes);
        if (bytes.length > 0 && !seen.has(key)) {
            seen.add(key);
            baseTokens.push(bytes);
        }
    }
}
const suffixTokens = endings
    .map((value) => new TextEncoder().encode(value))
    .filter((bytes) => !seen.has(String.fromCharCode(...bytes)));

function variant(extra: Uint8Array[]): LosslessByteTokenizer {
    const tokenizer = new LosslessByteTokenizer(baseTokens.length + extra.length);
    for (const token of baseTokens) tokenizer.add(token);
    for (const token of extra) tokenizer.add(token);
    return tokenizer;
}

const variants: Array<{ name: string; tokenizer: TokenizerLike }> = [
    { name: "bpe-2048", tokenizer: variant([]) },
    { name: "bpe-2048-suffix", tokenizer: variant(suffixTokens) },
    { name: "caracter", tokenizer: buildTokenizer(programFromGenome([1, 0, 0, 0]), train) },
];

function corpus(tokenizer: TokenizerLike, name: string): void {
    mkdirSync("data/gpu", { recursive: true });
    writeFileSync(
        `data/gpu/${name}.json`,
        `${JSON.stringify({
            name,
            vocabulary: tokenizer.vocabularySize,
            exact: tokenizer.decode(tokenizer.encode(train)) === train &&
                tokenizer.decode(tokenizer.encode(test)) === test &&
                tokenizer.decode(tokenizer.encode(external)) === external,
            trainTokens: tokenizer.encode(train),
            testTokens: tokenizer.encode(test),
            externalTokens: tokenizer.encode(external),
            bytes: {
                train: Buffer.byteLength(train),
                test: Buffer.byteLength(test),
                external: Buffer.byteLength(external),
            },
        })}\n`,
        "utf8",
    );
}

function probe(tokenizer: TokenizerLike, name: string): void {
    const records = [];
    for (const record of reference.records) {
        const start = record.sentence.indexOf(record.form);
        if (start >= 0) {
            const target = start + record.form.length;
            const tokens = tokenizer.encode(record.sentence);
            const pieces = tokens.map((token) => tokenizer.decode([token]));
            let offset = 0;
            let position = -1;
            let index = 0;
            while (index < pieces.length && position < 0) {
                offset += (pieces[index] as string).length;
                if (offset >= target) position = index;
                index += 1;
            }
            if (position >= 0) {
                records.push({
                    form: record.form,
                    source: record.source,
                    tense: record.tense,
                    sentence: record.sentence,
                    tokens,
                    position,
                });
            }
        }
    }
    writeFileSync(`data/gpu/probe-${name}.json`, `${JSON.stringify(records)}\n`, "utf8");
    console.log(`${name}: ${records.length} frases`);
}

for (const { name, tokenizer } of variants) {
    corpus(tokenizer, name);
    probe(tokenizer, name);
}
