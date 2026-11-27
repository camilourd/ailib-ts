// Teoría: docs/teoria/nucleo/tokenizadores.md
// Experimento: byte, carácter, BPE 512/1024/2048 y palabra con el mismo LM.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { BPETokenizer } from "../../src/data/bpe.ts";
import { loadBPE } from "../../src/data/bpe-io.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharTokenizer } from "../../src/data/char.ts";
import { WordTokenizer } from "../../src/data/word.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";

interface Tokenizer {
    readonly vocabularySize: number;
    encode(text: string): number[];
    decode(tokens: readonly number[]): string;
}

const text = readFileSync("data/quijote.txt", "utf8");
const split = Math.floor(text.length * 0.9);
const trainText = text.slice(0, split);
const validationText = text.slice(split);
const trainChars = [...trainText].length;
const validationChars = [...validationText].length;
const trainBytes = Buffer.byteLength(trainText, "utf8");
const validationBytes = Buffer.byteLength(validationText, "utf8");

const tokenizers: Array<{ name: string; tokenizer: Tokenizer }> = [
    { name: "byte", tokenizer: new ByteTokenizer() },
    { name: "carácter", tokenizer: new CharTokenizer().train(trainText) },
    { name: "bpe-512", tokenizer: loadBPE("data/quijote-bpe.json") },
];
for (const vocabularySize of [1024, 2048]) {
    const bpe = new BPETokenizer();
    bpe.train(trainText, { vocabularySize, minFrequency: 2 });
    tokenizers.push({ name: `bpe-${vocabularySize}`, tokenizer: bpe });
}
tokenizers.push({
    name: "palabra",
    tokenizer: new WordTokenizer().train(trainText, { minFrequency: 2 }),
});

const context = 96;
const steps = 600;
const learningRate = 0.005;
const results: Array<Record<string, number | string>> = [];

for (const { name, tokenizer } of tokenizers) {
    const trainTokens = tokenizer.encode(trainText);
    const validationTokens = tokenizer.encode(validationText);
    const model = new TransformerLM(tokenizer.vocabularySize, {
        modelSize: 64,
        heads: 4,
        blocks: 2,
        hiddenSize: 256,
        seed: 1,
    });
    const optimizer = new SGD(model.parameters(), learningRate);
    const rng = createRng(1);
    for (let step = 0; step < steps; step++) {
        const start = rng.int(trainTokens.length - context - 1);
        const piece = trainTokens.slice(start, start + context + 1);
        optimizer.zeroGrad();
        const loss = model.loss(piece);
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }

    const stride = Math.max(1, Math.floor((validationTokens.length - context - 1) / 4));
    let total = 0;
    for (let window = 0; window < 4; window++) {
        const start = window * stride;
        total += model.loss(validationTokens.slice(start, start + context + 1)).get();
    }
    const validationLoss = total / 4;
    const bitsPerByte = (validationLoss * validationTokens.length) / validationBytes / Math.LN2;
    const bitsPerChar = (validationLoss * validationTokens.length) / validationChars / Math.LN2;
    results.push({
        name,
        vocabulary: tokenizer.vocabularySize,
        tokensPerChar: trainTokens.length / trainChars,
        bitsPerByte,
        bitsPerChar,
        perplexity: Math.exp(validationLoss),
        parameters: model.parameterCount(),
    });
    console.log(
        `${name.padEnd(10)} vocab ${String(tokenizer.vocabularySize).padStart(5)} · ` +
            `${(trainTokens.length / trainChars).toFixed(3)} tokens/carácter · ` +
            `BPB ${bitsPerByte.toFixed(3)} · ` +
            `bits/carácter ${bitsPerChar.toFixed(3)} · ppl ${Math.exp(validationLoss).toFixed(1)}`,
    );
}

writeFileSync(
    "data/tokenizers-report.json",
    `${JSON.stringify(
        {
            data: { trainChars, validationChars, trainBytes, validationBytes },
            model: { modelSize: 64, heads: 4, blocks: 2, hiddenSize: 256 },
            steps,
            learningRate,
            context,
            results,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/tokenizers-report.json");
