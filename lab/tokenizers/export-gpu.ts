// F9.4b: exporta tokens para el entrenador GPU.
// Base BPE 2048 y variante BPE + sufijos verbales; ambas lossless y con el mismo tokenizador base.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { ContextTokenizer } from "../../src/data/context-tokenizer.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const testStart = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const test = quijote.slice(testStart);
const externalText = external;

const endings = [
    "ábamos", "ábais", "abais", "aban", "abas", "aba",
    "íamos", "íais", "ían", "ías", "ía",
    "ieron", "ió", "aste", "aron", "ado", "ido", "ando", "iendo",
    "aremos", "aréis", "arán", "ará", "eremos", "eréis", "erán", "erá",
    "iremos", "iréis", "irán", "irá", "aría", "arías", "arían",
    "amos", "áis", "an", "as", "en", "es", "emos", "éis",
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
const suffixTokens = endings
    .map((suffix) => new TextEncoder().encode(suffix))
    .filter((bytes) => !seen.has(String.fromCharCode(...bytes)));

function exportVariant(name: string, withSuffixes: boolean): void {
    const tokens = withSuffixes ? [...baseTokens, ...suffixTokens] : baseTokens;
    const tokenizer = new ContextTokenizer(tokens, {
        leftContext: 0,
        rightContext: 0,
        lookahead: "chars",
        penalty: 0,
    });
    const trainTokens = tokenizer.encode(train);
    const testTokens = tokenizer.encode(test);
    const externalTokens = tokenizer.encode(externalText);
    const exact =
        tokenizer.decode(trainTokens) === train &&
        tokenizer.decode(testTokens) === test &&
        tokenizer.decode(externalTokens) === externalText;
    const payload = {
        name,
        vocabulary: tokenizer.vocabularySize,
        exact,
        trainTokens,
        testTokens,
        externalTokens,
        bytes: {
            train: Buffer.byteLength(train, "utf8"),
            test: Buffer.byteLength(test, "utf8"),
            external: Buffer.byteLength(externalText, "utf8"),
        },
        characters: {
            train: [...train].length,
            test: [...test].length,
            external: [...externalText].length,
        },
    };
    mkdirSync("data/gpu", { recursive: true });
    writeFileSync(`data/gpu/${name}.json`, `${JSON.stringify(payload)}\n`, "utf8");
    console.log(
        `${name}: vocab ${tokenizer.vocabularySize} · train ${trainTokens.length} · ` +
            `test ${testTokens.length} · externo ${externalTokens.length} · exacto ${exact}`,
    );
}

exportVariant("bpe-2048", false);
exportVariant("bpe-2048-suffix", true);
