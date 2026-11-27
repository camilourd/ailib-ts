// F12.2: exporta BPE 2k/8k y SentencePiece 8k por dominio para el LM.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { SentencePieceTokenizer } from "../../src/data/sentencepiece.ts";

interface TokenizerLike {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const domains = ["narrativo", "tecnico", "codigo"] as const;

function bpe(train: string, vocabularySize: number): TokenizerLike {
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

function save(name: string, tokenizer: TokenizerLike, train: string, validation: string, test: string): void {
    const trainTokens = tokenizer.encode(train);
    const validationTokens = tokenizer.encode(validation);
    const testTokens = tokenizer.encode(test);
    if (
        tokenizer.decode(trainTokens) !== train ||
        tokenizer.decode(validationTokens) !== validation ||
        tokenizer.decode(testTokens) !== test
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
            validationTokens,
            testTokens,
            externalTokens: validationTokens,
            bytes: {
                train: Buffer.byteLength(train),
                validation: Buffer.byteLength(validation),
                test: Buffer.byteLength(test),
                external: Buffer.byteLength(validation),
            },
        })}\n`,
        "utf8",
    );
    console.log(`${name}: vocab ${tokenizer.vocabularySize} · test ${testTokens.length}`);
}

for (const domain of domains) {
    const train = readFileSync(`data/corpora/${domain}-train.txt`, "utf8").slice(0, 50000);
    const validation = readFileSync(`data/corpora/${domain}-validation.txt`, "utf8");
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    const bpe2 = bpe(train, 2048);
    const bpe8 = bpe(train, 8192);
    const sentencepiece = new SentencePieceTokenizer();
    sentencepiece.train(train, { vocabularySize: 8192 });
    save(`${domain}-bpe-2k`, bpe2, train, validation, test);
    save(`${domain}-bpe-8k`, bpe8, train, validation, test);
    save(`${domain}-sentencepiece-8k`, sentencepiece, train, validation, test);
}
