// F14: mezcla con el train completo de los 3 dominios, tokenizada con BPE 2k.
import { readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

function build(train: string): TokenizerLike {
    const bpe = new BPETokenizer();
    bpe.train(train, { vocabularySize: 2048 });
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

const domains = ["narrativo", "tecnico", "codigo"] as const;
const trainTokens: number[] = [];
const testSets: Record<string, number[]> = {};
const validationSets: Record<string, number[]> = {};
const bytes: Record<string, number> = {};

for (const domain of domains) {
    const source = readFileSync(`data/corpora/${domain}-train.txt`, "utf8");
    const tokenizer = build(source.slice(0, 50000));
    const train = source;
    const validation = readFileSync(`data/corpora/${domain}-validation.txt`, "utf8");
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    const encodedTrain = tokenizer.encode(train);
    const encodedValidation = tokenizer.encode(validation);
    const encodedTest = tokenizer.encode(test);
    if (
        tokenizer.decode(encodedTrain) !== train ||
        tokenizer.decode(encodedValidation) !== validation ||
        tokenizer.decode(encodedTest) !== test
    ) {
        throw new Error(`${domain} no es lossless`);
    }
    trainTokens.push(...encodedTrain);
    validationSets[domain] = encodedValidation;
    testSets[domain] = encodedTest;
    bytes[domain] = Buffer.byteLength(test);
}
bytes.test = bytes.narrativo as number;
bytes.external = bytes.narrativo as number;

writeFileSync(
    "data/gpu/f14-mixture-bpe-2k.json",
    `${JSON.stringify({
        name: "f14-mixture-bpe-2k",
        vocabulary: 2048,
        exact: true,
        trainTokens,
        testTokens: testSets.narrativo,
        externalTokens: validationSets.narrativo,
        testSets,
        validationSets,
        bytes,
    })}\n`,
    "utf8",
);
console.log(`train ${trainTokens.length} tokens; ${Object.entries(testSets).map(([name, tokens]) => `${name} ${tokens.length}`).join(" · ")}`);
