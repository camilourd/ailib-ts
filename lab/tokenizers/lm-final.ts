// F9.4: LM 1.4M con el tokenizer ganador frente a BPE 2048, palabra y carácter.
// Fitness final: BPB en Quijote test y Lazarillo externo.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { loadBPE } from "../../src/data/bpe-io.ts";
import { buildTokenizer, type ProgramTokenizer, programFromGenome } from "../../src/data/tokenizer-program.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const testStart = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const test = quijote.slice(testStart);
const externalText = external;

const tokenizers: Array<{ name: string; tokenizer: ProgramTokenizer }> = [
    {
        name: "substring-4k",
        tokenizer: buildTokenizer(programFromGenome([3, 3, 0, 0]), train),
    },
    { name: "bpe-2048", tokenizer: loadBPE("data/quijote-bpe.json") },
    { name: "palabra", tokenizer: buildTokenizer(programFromGenome([2, 2, 0, 1]), train) },
    { name: "carácter", tokenizer: buildTokenizer(programFromGenome([1, 0, 0, 0]), train) },
];

const context = 128;
const steps = 150;
const learningRate = 0.003;
const results: Array<Record<string, number | string>> = [];

for (const { name, tokenizer } of tokenizers) {
    console.log(`construido ${name}: vocab ${tokenizer.vocabularySize}`);
    const trainTokens = tokenizer.encode(train);
    const testTokens = tokenizer.encode(test);
    const externalTokens = tokenizer.encode(externalText);
    const model = new TransformerLM(tokenizer.vocabularySize, {
        modelSize: 96,
        heads: 6,
        blocks: 2,
        hiddenSize: 384,
        seed: 1,
    });
    console.log(`entrenando ${name}: ${model.parameterCount()} parámetros`);
    const optimizer = new SGD(model.parameters(), learningRate);
    const rng = createRng(1);
    for (let step = 0; step < steps; step++) {
        const start = rng.int(trainTokens.length - context - 1);
        optimizer.zeroGrad();
        const loss = model.loss(trainTokens.slice(start, start + context + 1));
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }
    const measure = (tokens: number[], characters: number): number => {
        const stride = Math.max(1, Math.floor((tokens.length - context - 1) / 8));
        let total = 0;
        for (let window = 0; window < 8; window++) {
            const start = window * stride;
            total += model.loss(tokens.slice(start, start + context + 1)).get();
        }
        const mean = total / 8;
        return (mean * tokens.length) / characters / Math.LN2;
    };
    const bpbTest = measure(testTokens, Buffer.byteLength(test, "utf8"));
    const bpbExternal = measure(externalTokens, Buffer.byteLength(externalText, "utf8"));
    const row = {
        name,
        vocabulary: tokenizer.vocabularySize,
        parameters: model.parameterCount(),
        trainTokens: trainTokens.length,
        testTokens: testTokens.length,
        bpbTest,
        bpbExternal,
    };
    results.push(row);
    console.log(
        `${name.padEnd(13)} vocab ${String(tokenizer.vocabularySize).padStart(5)} · ` +
            `params ${String(model.parameterCount()).padStart(8)} · ` +
            `BPB test ${bpbTest.toFixed(3)} · externo ${bpbExternal.toFixed(3)}`,
    );
    writeFileSync(
        "data/tokenizer-lm-final.json",
        `${JSON.stringify({ context, steps, learningRate, results }, null, 2)}\n`,
        "utf8",
    );
}
console.log("guardado: data/tokenizer-lm-final.json");
