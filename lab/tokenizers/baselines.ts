// F9.0: baselines de tokenización lossless sobre Quijote y Lazarillo.
// La comparación principal es tokens/carácter; el decode exacto es una restricción.

import { readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { loadBPE } from "../../src/data/bpe-io.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharTokenizer } from "../../src/data/char.ts";
import { WordTokenizer } from "../../src/data/word.ts";

interface Tokenizer {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const validationEnd = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const validation = quijote.slice(trainEnd, validationEnd);
const test = quijote.slice(validationEnd);

const tokenizers: Array<{ name: string; tokenizer: Tokenizer }> = [
    { name: "byte", tokenizer: new ByteTokenizer() },
    { name: "carácter", tokenizer: new CharTokenizer().train(train) },
    { name: "bpe-512", tokenizer: loadBPE("data/quijote-bpe.json") },
];
for (const vocabularySize of [1024, 2048]) {
    const bpe = new BPETokenizer();
    bpe.train(train, { vocabularySize, minFrequency: 2 });
    tokenizers.push({ name: `bpe-${vocabularySize}`, tokenizer: bpe });
}
tokenizers.push({ name: "palabra", tokenizer: new WordTokenizer().train(train, { minFrequency: 2 }) });

const chars = (text: string): number => [...text].length;
const rows = [];
for (const { name, tokenizer } of tokenizers) {
    const parts = { train, validation, test, external } as const;
    const metrics: Record<string, unknown> = {
        name,
        vocabulary: tokenizer.vocabularySize,
    };
    let allExact = true;
    for (const [part, text] of Object.entries(parts)) {
        const tokens = tokenizer.encode(text);
        const exact = tokenizer.decode(tokens) === text;
        allExact &&= exact;
        metrics[part] = {
            characters: chars(text),
            bytes: Buffer.byteLength(text, "utf8"),
            tokens: tokens.length,
            tokensPerChar: tokens.length / chars(text),
            exact,
        };
    }
    metrics.exact = allExact;
    rows.push(metrics);
    console.log(
        `${name.padEnd(10)} vocab ${String(tokenizer.vocabularySize).padStart(5)} · ` +
            `val ${((metrics.validation as { tokensPerChar: number }).tokensPerChar).toFixed(3)} tokens/carácter · ` +
            `externo ${((metrics.external as { tokensPerChar: number }).tokensPerChar).toFixed(3)} · ` +
            `exacto ${allExact ? "sí" : "no"}`,
    );
}

writeFileSync(
    "data/tokenizer-baselines.json",
    `${JSON.stringify(
        {
            corpus: {
                quijote: { train: chars(train), validation: chars(validation), test: chars(test) },
                external: { name: "Lazarillo de Tormes", characters: chars(external) },
            },
            rows,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-baselines.json");
