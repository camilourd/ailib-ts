// F10.1: arnés de métricas multidimensionales para tokenizers.
// Exactitud, compresión, fronteras morfológicas, ED/BLEU, NUT/LD y BPB proxy.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { buildTokenizer, programFromGenome } from "../../src/data/tokenizer-program.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

interface Reference {
    form: string;
    stem: string;
    suffix: string;
    tense: string;
    confidence: string;
}

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const testStart = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const test = quijote.slice(testStart);
const externalText = external;
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
const prefixes = [
    "des", "re", "in", "im", "pre", "sub", "inter", "trans", "anti",
    "auto", "co", "ex", "extra", "hiper", "micro", "multi", "post",
    "super", "ultra",
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
const extras = (values: string[]): Uint8Array[] =>
    values
        .map((value) => new TextEncoder().encode(value))
        .filter((bytes) => !seen.has(String.fromCharCode(...bytes)));

function baseVariant(extraTokens: Uint8Array[]): TokenizerLike {
    const tokenizer = new LosslessByteTokenizer(baseTokens.length + extraTokens.length);
    for (const token of baseTokens) tokenizer.add(token);
    for (const token of extraTokens) tokenizer.add(token);
    return tokenizer;
}

const variants: Array<{ name: string; tokenizer: TokenizerLike }> = [
    { name: "bpe-2048", tokenizer: baseVariant([]) },
    { name: "bpe-2048-suffix", tokenizer: baseVariant(extras(endings)) },
    { name: "bpe-2048-prefix-suffix", tokenizer: baseVariant(extras([...prefixes, ...endings])) },
    { name: "palabra", tokenizer: buildTokenizer(programFromGenome([2, 2, 0, 1]), train) },
    { name: "caracter", tokenizer: buildTokenizer(programFromGenome([1, 0, 0, 0]), train) },
];

const sentences = test.split(/(?<=[.!?…])\s+/u).filter((sentence) => sentence.trim().length > 0);

function tokenizeIds(tokenizer: TokenizerLike, text: string): string[] {
    return tokenizer.encode(text).map((token) => tokenizer.decode([token]));
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

function ngrams(tokens: string[], size: number): Map<string, number> {
    const counts = new Map<string, number>();
    for (let index = 0; index + size <= tokens.length; index++) {
        const key = tokens.slice(index, index + size).join("\u0001");
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return counts;
}

function bleu(referenceTokens: string[], predictedTokens: string[], size = 2): number {
    if (predictedTokens.length === 0) return 0;
    let logSum = 0;
    for (let order = 1; order <= size; order++) {
        const predicted = ngrams(predictedTokens, order);
        const referenceCounts = ngrams(referenceTokens, order);
        let overlap = 0;
        let total = 0;
        for (const [gram, count] of predicted) {
            overlap += Math.min(count, referenceCounts.get(gram) ?? 0);
            total += count;
        }
        if (total === 0 || overlap === 0) return 0;
        logSum += Math.log(overlap / total);
    }
    const brevity = predictedTokens.length < referenceTokens.length
        ? Math.exp(1 - referenceTokens.length / predictedTokens.length)
        : 1;
    return brevity * Math.exp(logSum / size);
}

function boundaryMetrics(tokenizer: TokenizerLike): { precision: number; recall: number; f1: number; support: number } {
    let truePositives = 0;
    let predictedTotal = 0;
    let support = 0;
    for (const record of reference.records) {
        if (record.confidence === "alta" && record.stem && record.suffix && record.stem.length < record.form.length) {
            support += 1;
            const pieces = tokenizer.encode(record.form).map((token) => tokenizer.decode([token]));
            const boundaries = new Set<number>();
            let offset = 0;
            for (let index = 0; index < pieces.length - 1; index++) {
                offset += (pieces[index] as string).length;
                boundaries.add(offset);
            }
            predictedTotal += boundaries.size;
            if (boundaries.has(record.stem.length)) truePositives += 1;
        }
    }
    const precision = predictedTotal === 0 ? 0 : truePositives / predictedTotal;
    const recall = support === 0 ? 0 : truePositives / support;
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return { precision, recall, f1, support };
}

function diversity(tokenizer: TokenizerLike): { nut: number; ld: number } {
    let unique = 0;
    let total = 0;
    for (const sentence of sentences) {
        const tokens = tokenizer.encode(sentence);
        unique += new Set(tokens).size;
        total += tokens.length;
    }
    return { nut: unique / sentences.length, ld: total === 0 ? 0 : unique / total };
}

function proxyBpb(tokenizer: TokenizerLike): number {
    const model = new TransformerLM(tokenizer.vocabularySize, {
        modelSize: 32, heads: 4, blocks: 1, hiddenSize: 64, seed: 1,
    });
    const optimizer = new SGD(model.parameters(), 0.005);
    const trainTokens = tokenizer.encode(train);
    const testTokens = tokenizer.encode(test);
    const context = 64;
    const rng = createRng(1);
    for (let step = 0; step < 200; step++) {
        const start = rng.int(trainTokens.length - context - 1);
        optimizer.zeroGrad();
        const loss = model.loss(trainTokens.slice(start, start + context + 1));
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }
    const stride = Math.max(1, Math.floor((testTokens.length - context - 1) / 4));
    let total = 0;
    for (let window = 0; window < 4; window++) {
        total += model.loss(testTokens.slice(window * stride, window * stride + context + 1)).get();
    }
    return ((total / 4) * testTokens.length) / Buffer.byteLength(test, "utf8") / Math.LN2;
}

const rows = [];
for (const variant of variants) {
    const tokenizer = variant.tokenizer;
    const trainTokens = tokenizer.encode(train);
    const testTokens = tokenizer.encode(test);
    const externalTokens = tokenizer.encode(externalText);
    const exact =
        tokenizer.decode(trainTokens) === train &&
        tokenizer.decode(testTokens) === test &&
        tokenizer.decode(externalTokens) === externalText;
    const boundary = boundaryMetrics(tokenizer);
    const diversityMetrics = diversity(tokenizer);
    let distance = 0;
    let bleuTotal = 0;
    let count = 0;
    for (const record of reference.records) {
        if (record.confidence === "alta" && record.stem && record.suffix) {
            const predicted = tokenizeIds(tokenizer, record.form);
            const expected = [record.stem, record.suffix];
            distance += editDistance(expected, predicted) / Math.max(expected.length, predicted.length);
            bleuTotal += bleu(expected, predicted);
            count += 1;
        }
    }
    rows.push({
        name: variant.name,
        vocabulary: tokenizer.vocabularySize,
        exact,
        tokensPerChar: testTokens.length / [...test].length,
        compressionRatio: [...test].length / testTokens.length,
        ant: testTokens.length / sentences.length,
        boundary,
        editDistance: count === 0 ? 0 : distance / count,
        bleu: count === 0 ? 0 : bleuTotal / count,
        diversity: diversityMetrics,
        bpbProxy: proxyBpb(tokenizer),
    });
    const row = rows[rows.length - 1] as (typeof rows)[number];
    console.log(
        `${variant.name.padEnd(24)} vocab ${String(row.vocabulary).padStart(5)} · ` +
            `tokens/car ${row.tokensPerChar.toFixed(3)} · frontera F1 ${row.boundary.f1.toFixed(3)} · ` +
            `ED ${row.editDistance.toFixed(3)} · BLEU ${row.bleu.toFixed(3)} · BPB ${row.bpbProxy.toFixed(3)}`,
    );
}

writeFileSync(
    "data/tokenizer-benchmark.json",
    `${JSON.stringify({ protocol: { train: [...train].length, test: [...test].length, external: [...externalText].length, sentences: sentences.length }, rows }, null, 2)}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-benchmark.json");
