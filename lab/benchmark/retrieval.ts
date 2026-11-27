// F12.3: retrieval de la unidad siguiente con BM25 sobre tokens.
// Es CPU y mide si el tokenizer conserva diferencias léxicas, no solo compresión.

import { readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { SentencePieceTokenizer } from "../../src/data/sentencepiece.ts";

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const domains = ["narrativo", "tecnico", "codigo"] as const;
const endings = ["ando", "iendo", "ado", "ido", "amos", "emos", "imos", "aba", "ía", "ieron", "ió", "an", "en", "es", "as", "os"];
const prefixes = ["des", "re", "in", "pre", "sub", "inter", "trans", "anti", "auto", "co", "ex", "super", "ultra"];
const k1 = 1.5;
const b = 0.75;

function bpe(train: string, vocabularySize: number, extras: readonly Uint8Array[] = []): TokenizerLike {
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
    for (const extra of extras) {
        const key = String.fromCharCode(...extra);
        if (!seen.has(key)) {
            seen.add(key);
            tokens.push(extra);
        }
    }
    const tokenizer = new LosslessByteTokenizer(tokens.length);
    for (const token of tokens) tokenizer.add(token);
    return tokenizer;
}

function units(text: string, domain: string): string[] {
    if (domain === "codigo") {
        return text.split("\n").map((line) => line.trim()).filter((line) => line.length >= 20 && line.length <= 240);
    }
    return text
        .split(/(?<=[.!?])\s+/)
        .map((sentence) => sentence.trim())
        .filter((sentence) => sentence.length >= 30 && sentence.length <= 300);
}

function score(query: readonly number[], documents: readonly (readonly number[])[]): number[] {
    const documentFrequency = new Map<number, number>();
    const lengths = documents.map((document) => document.length);
    const average = lengths.reduce((total, value) => total + value, 0) / Math.max(1, lengths.length);
    for (const document of documents) {
        for (const token of new Set(document)) {
            documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
        }
    }
    return documents.map((document, index) => {
        const frequencies = new Map<number, number>();
        for (const token of document) frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
        let value = 0;
        for (const token of query) {
            const frequency = frequencies.get(token) ?? 0;
            if (frequency !== 0) {
                const df = documentFrequency.get(token) ?? 0;
                const idf = Math.log(1 + (documents.length - df + 0.5) / (df + 0.5));
                const length = lengths[index] as number;
                value += idf * ((frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * (length / average))));
            }
        }
        return value;
    });
}

const rows: Array<Record<string, number | string>> = [];
for (const domain of domains) {
    const train = readFileSync(`data/corpora/${domain}-train.txt`, "utf8").slice(0, 50000);
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    const all = units(test, domain);
    const limit = Math.min(300, all.length - 1);
    const documents = all.slice(0, limit);
    const queries = all.slice(1, limit + 1);
    const variants: Array<[string, () => TokenizerLike]> = [
        ["bpe-2k", () => bpe(train, 2048)],
        ["bpe-2k-prefix-suffix", () => bpe(train, 2048, [
            ...prefixes.map((value) => new TextEncoder().encode(value)),
            ...endings.map((value) => new TextEncoder().encode(value)),
        ])],
        ["bpe-8k", () => bpe(train, 8192)],
        ["sentencepiece-8k", () => {
            const tokenizer = new SentencePieceTokenizer();
            tokenizer.train(train, { vocabularySize: 8192 });
            return tokenizer;
        }],
    ];
    for (const [name, build] of variants) {
        const tokenizer = build();
        const index = documents.map((document) => tokenizer.encode(document));
        let recall1 = 0;
        let recall5 = 0;
        let reciprocal = 0;
        for (let queryIndex = 0; queryIndex < queries.length; queryIndex++) {
            const query = tokenizer.encode(queries[queryIndex] as string);
            const values = score(query, index);
            const ranked = values
                .map((value, position) => ({ value, position }))
                .sort((left, right) => right.value - left.value || left.position - right.position);
            const rank = ranked.findIndex((entry) => entry.position === queryIndex) + 1;
            if (rank === 1) recall1 += 1;
            if (rank > 0 && rank <= 5) recall5 += 1;
            if (rank > 0) reciprocal += 1 / rank;
        }
        const total = queries.length;
        rows.push({
            domain,
            tokenizer: name,
            vocabulary: tokenizer.vocabularySize,
            pairs: total,
            recall1: recall1 / total,
            recall5: recall5 / total,
            mrr: reciprocal / total,
        });
        console.log(
            `${domain.padEnd(9)} ${name.padEnd(22)} vocab ${String(tokenizer.vocabularySize).padStart(5)} · ` +
                `R@1 ${(recall1 / total).toFixed(3)} · R@5 ${(recall5 / total).toFixed(3)} · MRR ${(reciprocal / total).toFixed(3)}`,
        );
        writeFileSync("data/tokenizer-retrieval.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
    }
}
console.log("guardado: data/tokenizer-retrieval.json");
