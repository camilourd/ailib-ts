// F9.4b: variantes de forma de palabra, mayúsculas y BPE dropout.
// Exporta tokens para el entrenador GPU; todas las variantes son lossless.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { ContextTokenizer } from "../../src/data/context-tokenizer.ts";
import { createRng } from "../../src/core/random/index.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const testStart = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const test = quijote.slice(testStart);
const externalText = external;

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

const options = { leftContext: 0, rightContext: 0, lookahead: "chars" as const, penalty: 0 };
const base = new ContextTokenizer(baseTokens, options);
const capitalCounts = new Map<string, number>();
for (const word of train.match(/\p{Lu}[\p{L}]+/gu) ?? []) {
    capitalCounts.set(word, (capitalCounts.get(word) ?? 0) + 1);
}
const capitalTokens = [...capitalCounts.entries()]
    .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
    .slice(0, 64)
    .map(([word]) => new TextEncoder().encode(word))
    .filter((bytes) => !seen.has(String.fromCharCode(...bytes)));

interface Payload {
    name: string;
    vocabulary: number;
    exact: boolean;
    trainTokens: number[];
    testTokens: number[];
    externalTokens: number[];
    bytes: { train: number; test: number; external: number };
}

function save(name: string, trainTokens: number[], testTokens: number[], externalTokens: number[], vocabulary: number, exact: boolean): void {
    const payload: Payload = {
        name,
        vocabulary,
        exact,
        trainTokens,
        testTokens,
        externalTokens,
        bytes: {
            train: Buffer.byteLength(train, "utf8"),
            test: Buffer.byteLength(test, "utf8"),
            external: Buffer.byteLength(externalText, "utf8"),
        },
    };
    mkdirSync("data/gpu", { recursive: true });
    writeFileSync(`data/gpu/${name}.json`, `${JSON.stringify(payload)}\n`, "utf8");
    console.log(
        `${name}: vocab ${vocabulary} · train ${trainTokens.length} · test ${testTokens.length} · externo ${externalTokens.length} · exacto ${exact}`,
    );
}

function capitalVariant(): void {
    const tokenizer = new ContextTokenizer([...baseTokens, ...capitalTokens], options);
    const trainTokens = tokenizer.encode(train);
    const testTokens = tokenizer.encode(test);
    const externalTokens = tokenizer.encode(externalText);
    save(
        "bpe-2048-capital",
        trainTokens,
        testTokens,
        externalTokens,
        tokenizer.vocabularySize,
        tokenizer.decode(trainTokens) === train &&
            tokenizer.decode(testTokens) === test &&
            tokenizer.decode(externalTokens) === externalText,
    );
}

function shapeVariant(): void {
    const baseSize = base.vocabularySize;
    const encode = (text: string): number[] => {
        const tokens = base.encode(text);
        const output: number[] = [];
        for (const token of tokens) {
            const piece = base.decode([token]);
            const first = piece[0] ?? " ";
            if (first >= "A" && first <= "Z") {
                const upper = piece === piece.toLocaleUpperCase() && piece.length > 1;
                output.push(baseSize + (upper ? 1 : 0), token);
            } else {
                output.push(token);
            }
        }
        return output;
    };
    const decode = (tokens: readonly number[]): string => base.decode(tokens.filter((token) => token < baseSize));
    const trainTokens = encode(train);
    const testTokens = encode(test);
    const externalTokens = encode(externalText);
    save(
        "bpe-2048-shape",
        trainTokens,
        testTokens,
        externalTokens,
        baseSize + 2,
        decode(trainTokens) === train && decode(testTokens) === test && decode(externalTokens) === externalText,
    );
}

function dropoutVariant(): void {
    const rng = createRng(7);
    const trainTokens = base.encode(train).flatMap((token) => {
        if (rng.next() >= 0.1 || token <= 255) return [token];
        const piece = baseTokens[token] as Uint8Array;
        return [...piece];
    });
    const testTokens = base.encode(test);
    const externalTokens = base.encode(externalText);
    save(
        "bpe-2048-dropout",
        trainTokens,
        testTokens,
        externalTokens,
        base.vocabularySize,
        base.decode(trainTokens) === train &&
            base.decode(testTokens) === test &&
            base.decode(externalTokens) === externalText,
    );
}

capitalVariant();
shapeVariant();
dropoutVariant();
