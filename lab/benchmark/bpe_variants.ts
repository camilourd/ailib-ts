// F12.4: variantes de BPE sin LM: vocabulario, mezcla de dominios y dropout.
import { readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const domains = ["narrativo", "tecnico", "codigo"] as const;
const vocabularies = [1024, 1536, 2048, 3072, 4096];

function build(train: string, vocabularySize: number): TokenizerLike {
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

const rows: Array<Record<string, number | string | boolean>> = [];
for (const domain of domains) {
    const train = readFileSync(`data/corpora/${domain}-train.txt`, "utf8").slice(0, 50000);
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    for (const vocabularySize of vocabularies) {
        const tokenizer = build(train, vocabularySize);
        const tokens = tokenizer.encode(test);
        rows.push({
            domain,
            variant: `vocab-${vocabularySize}`,
            requested: vocabularySize,
            actual: tokenizer.vocabularySize,
            tokensPerChar: tokens.length / [...test].length,
            exact: tokenizer.decode(tokens) === test,
        });
        console.log(`${domain} vocab ${vocabularySize}: real ${tokenizer.vocabularySize} · ${(tokens.length / [...test].length).toFixed(3)} tokens/car`);
        writeFileSync("data/tokenizer-bpe-variants.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
    }
}

const mixture = domains
    .map((domain) => readFileSync(`data/corpora/${domain}-train.txt`, "utf8").slice(0, 50000))
    .join("\n");
const cross = build(mixture, 2048);
for (const domain of domains) {
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    const tokens = cross.encode(test);
    rows.push({
        domain,
        variant: "mezcla-2048",
        requested: 2048,
        actual: cross.vocabularySize,
        tokensPerChar: tokens.length / [...test].length,
        exact: cross.decode(tokens) === test,
    });
}
writeFileSync("data/tokenizer-bpe-variants.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
console.log("guardado: data/tokenizer-bpe-variants.json");
