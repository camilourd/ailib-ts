// F9.4 proxy rápido: comparar tokenizers con un LM pequeño; la corrida de 1M queda pendiente.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { BPETokenizer } from "../../src/data/bpe.ts";
import { buildTokenizer, programFromGenome, type ProgramTokenizer } from "../../src/data/tokenizer-program.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, 20000);
const test = quijote.slice(28000, 36000);
const externalText = external.slice(0, 10000);

const bpe = new BPETokenizer();
bpe.train(train, { vocabularySize: 2048, minFrequency: 2 });
const tokenizers: Array<{ name: string; tokenizer: ProgramTokenizer }> = [
    { name: "substring-4k", tokenizer: buildTokenizer(programFromGenome([3, 3, 0, 0]), train) },
    { name: "bpe-2048", tokenizer: bpe },
    { name: "palabra", tokenizer: buildTokenizer(programFromGenome([2, 2, 0, 1]), train) },
    { name: "carácter", tokenizer: buildTokenizer(programFromGenome([1, 0, 0, 0]), train) },
];

const context = 64;
const steps = 200;
const results = [];
for (const { name, tokenizer } of tokenizers) {
    const trainTokens = tokenizer.encode(train);
    const model = new TransformerLM(tokenizer.vocabularySize, {
        modelSize: 32,
        heads: 4,
        blocks: 1,
        hiddenSize: 64,
        seed: 1,
    });
    const optimizer = new SGD(model.parameters(), 0.005);
    const rng = createRng(1);
    for (let step = 0; step < steps; step++) {
        const start = rng.int(trainTokens.length - context - 1);
        optimizer.zeroGrad();
        const loss = model.loss(trainTokens.slice(start, start + context + 1));
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }
    const measure = (text: string): { bpb: number; tokens: number } => {
        const tokens = tokenizer.encode(text);
        const stride = Math.max(1, Math.floor((tokens.length - context - 1) / 4));
        let total = 0;
        for (let window = 0; window < 4; window++) {
            total += model.loss(tokens.slice(window * stride, window * stride + context + 1)).get();
        }
        return {
            bpb: ((total / 4) * tokens.length) / Buffer.byteLength(text, "utf8") / Math.LN2,
            tokens: tokens.length,
        };
    };
    const testResult = measure(test);
    const externalResult = measure(externalText);
    results.push({
        name,
        vocabulary: tokenizer.vocabularySize,
        parameters: model.parameterCount(),
        bpbTest: testResult.bpb,
        bpbExternal: externalResult.bpb,
        tokensPerCharTest: testResult.tokens / [...test].length,
    });
    console.log(
        `${name.padEnd(13)} vocab ${String(tokenizer.vocabularySize).padStart(5)} · ` +
            `BPB test ${testResult.bpb.toFixed(3)} · externo ${externalResult.bpb.toFixed(3)}`,
    );
}
writeFileSync("data/tokenizer-lm-quick.json", `${JSON.stringify({ context, steps, results }, null, 2)}\n`, "utf8");
console.log("guardado: data/tokenizer-lm-quick.json");
