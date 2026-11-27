// F12.5: BPE con pre-tokenización regex vs BPE normal.
import { readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { PreTokenizedBPETokenizer } from "../../src/data/pretokenized-bpe.ts";

function bpe(train: string): LosslessByteTokenizer {
    const raw = new BPETokenizer();
    raw.train(train, { vocabularySize: 2048 });
    const tokens: Uint8Array[] = [];
    const seen = new Set<string>();
    for (let byte = 0; byte < 256; byte++) {
        tokens.push(Uint8Array.from([byte]));
        seen.add(String.fromCharCode(byte));
    }
    for (let id = 0; id < raw.vocabularySize; id++) {
        const text = raw.decode([id]);
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
for (const domain of ["narrativo", "tecnico", "codigo"] as const) {
    const train = readFileSync(`data/corpora/${domain}-train.txt`, "utf8").slice(0, 50000);
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    const plain = bpe(train);
    const regex = new PreTokenizedBPETokenizer();
    regex.train(train, { vocabularySize: 2048 });
    const plainTokens = plain.encode(test);
    const regexTokens = regex.encode(test);
    rows.push({
        domain,
        plain: plainTokens.length / [...test].length,
        regex: regexTokens.length / [...test].length,
        regexExact: regex.decode(regexTokens) === test,
        regexVocabulary: regex.vocabularySize,
    });
    console.log(`${domain}: BPE ${(plainTokens.length / [...test].length).toFixed(3)} · regex ${(regexTokens.length / [...test].length).toFixed(3)}`);
}
writeFileSync("data/tokenizer-regex.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
