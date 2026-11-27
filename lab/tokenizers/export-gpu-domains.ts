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
const prepositions = [
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según",
    "sin", "so", "sobre", "tras", "versus", "vía",
];

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
    const bpe2Hybrid = bpe(train, 2048, [
        ...prefixes.map((value) => new TextEncoder().encode(value)),
        ...endings.map((value) => new TextEncoder().encode(value)),
    ]);
    const bpe2Prepositions = bpe(train, 2048, [
        ...prefixes.map((value) => new TextEncoder().encode(value)),
        ...endings.map((value) => new TextEncoder().encode(value)),
        ...prepositions.map((value) => new TextEncoder().encode(value)),
    ]);
    const bpe8 = bpe(train, 8192);
    const sentencepiece = new SentencePieceTokenizer();
    sentencepiece.train(train, { vocabularySize: 8192 });
    save(`${domain}-bpe-2k`, bpe2, train, validation, test);
    save(`${domain}-bpe-prefix-suffix`, bpe2Hybrid, train, validation, test);
    save(`${domain}-bpe-prefix-suffix-preposition`, bpe2Prepositions, train, validation, test);
    save(`${domain}-bpe-8k`, bpe8, train, validation, test);
    save(`${domain}-sentencepiece-8k`, sentencepiece, train, validation, test);
}
