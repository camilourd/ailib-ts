// F11.4: exportación rápida y justa a 2k sobre el mismo corpus de 50k.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { SentencePieceTokenizer } from "../../src/data/sentencepiece.ts";
import { UnigramTokenizer } from "../../src/data/unigram.ts";
import { WordPieceTokenizer } from "../../src/data/wordpiece.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, 50000);
const test = quijote.slice(Math.floor(quijote.length * 0.85));

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

function exportVariant(name: string, tokenizer: TokenizerLike): void {
    if (existsSync(`data/gpu/${name}.json`)) {
        console.log(`${name}: ya existe`);
        return;
    }
    const trainTokens = tokenizer.encode(train);
    const testTokens = tokenizer.encode(test);
    const externalTokens = tokenizer.encode(external);
    if (
        tokenizer.decode(trainTokens) !== train ||
        tokenizer.decode(testTokens) !== test ||
        tokenizer.decode(externalTokens) !== external
    ) {
        throw new Error(`${name} no es lossless`);
    }
    mkdirSync("data/gpu", { recursive: true });
    writeFileSync(
        `data/gpu/${name}.json`,
        `${JSON.stringify({
            name,
            vocabulary: tokenizer.vocabularySize,
            exact: true,
            trainTokens,
            testTokens,
            externalTokens,
            bytes: {
                train: Buffer.byteLength(train),
                test: Buffer.byteLength(test),
                external: Buffer.byteLength(external),
            },
        })}\n`,
        "utf8",
    );
    console.log(`${name}: vocab ${tokenizer.vocabularySize} · test ${testTokens.length}`);
}

function bpeTokens(): Uint8Array[] {
    const bpe = new BPETokenizer();
    bpe.train(train, { vocabularySize: 2048 });
    const tokens: Uint8Array[] = [];
    for (let byte = 0; byte < 256; byte++) tokens.push(Uint8Array.from([byte]));
    const seen = new Set<string>();
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
    return tokens;
}

function bpeWithExtras(extras: Uint8Array[]): LosslessByteTokenizer {
    const base = bpeTokens();
    const tokenizer = new LosslessByteTokenizer(base.length + extras.length);
    for (const token of base) tokenizer.add(token);
    for (const token of extras) tokenizer.add(token);
    return tokenizer;
}

const endings = [
    "ábamos", "ábais", "abais", "aban", "abas", "aba",
    "íamos", "íais", "ían", "ías", "ía",
    "ieron", "ió", "aste", "aron", "ado", "ido", "ando", "iendo",
    "aremos", "aréis", "arán", "ará", "eremos", "eréis", "erán", "erá",
    "iremos", "iréis", "irán", "irá", "aría", "arías", "arían",
    "amos", "áis", "an", "as", "en", "es", "emos", "éis",
];
const prefixes = [
    "des", "re", "in", "im", "pre", "sub", "inter", "trans", "anti",
    "auto", "co", "ex", "extra", "hiper", "micro", "multi", "post",
    "super", "ultra",
];
const extraBytes = (values: string[]): Uint8Array[] =>
    values.map((value) => new TextEncoder().encode(value));
const numberCounts = new Map<string, number>();
for (const match of train.matchAll(/\p{N}+/gu)) {
    numberCounts.set(match[0], (numberCounts.get(match[0]) ?? 0) + 1);
}
const numberTokens = [...numberCounts.entries()]
    .filter(([, count]) => count >= 2)
    .sort((first, second) => second[1] - first[1])
    .slice(0, 128)
    .map(([value]) => new TextEncoder().encode(value));

const sentencepiece = new SentencePieceTokenizer();
sentencepiece.train(train, { vocabularySize: 2048 });
const wordpiece = new WordPieceTokenizer();
wordpiece.train(train, { vocabularySize: 2048, minFrequency: 5 });
const unigram = new UnigramTokenizer();
unigram.train(train, { vocabularySize: 2048, iterations: 2, minFrequency: 2 });

exportVariant("fair-bpe-2048", bpeWithExtras([]));
exportVariant("fair-bpe-suffix", bpeWithExtras(extraBytes(endings)));
exportVariant("fair-bpe-prefix-suffix", bpeWithExtras(extraBytes([...prefixes, ...endings])));
exportVariant("fair-bpe-prefix-suffix-number", bpeWithExtras(extraBytes([...prefixes, ...endings]).concat(numberTokens)));
exportVariant("fair-sentencepiece-2048", sentencepiece);
exportVariant("fair-wordpiece-2048", wordpiece);
exportVariant("fair-unigram-2048", unigram);
