// F11.4: exportación rápida y justa a 2k sobre el mismo corpus de 50k.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

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

function bpe(): LosslessByteTokenizer {
    const bpe = new BPETokenizer();
    bpe.train(train, { vocabularySize: 2048 });
    const tokenizer = new LosslessByteTokenizer(2048);
    for (let byte = 0; byte < 256; byte++) tokenizer.add(Uint8Array.from([byte]));
    const seen = new Set<string>();
    for (let id = 0; id < bpe.vocabularySize; id++) {
        const text = bpe.decode([id]);
        if (text.includes("\uFFFD")) continue;
        const bytes = new TextEncoder().encode(text);
        const key = String.fromCharCode(...bytes);
        if (!seen.has(key)) {
            seen.add(key);
            tokenizer.add(bytes);
        }
    }
    return tokenizer;
}

const sentencepiece = new SentencePieceTokenizer();
sentencepiece.train(train, { vocabularySize: 2048 });
const wordpiece = new WordPieceTokenizer();
wordpiece.train(train, { vocabularySize: 2048, minFrequency: 5 });
const unigram = new UnigramTokenizer();
unigram.train(train, { vocabularySize: 2048, iterations: 2, minFrequency: 2 });

exportVariant("fair-bpe-2048", bpe());
exportVariant("fair-sentencepiece-2048", sentencepiece);
exportVariant("fair-wordpiece-2048", wordpiece);
exportVariant("fair-unigram-2048", unigram);
