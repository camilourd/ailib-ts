// F9.4b proxy: BPE vs BPE+sufijos con un LM pequeño; GPU queda pendiente.
import { readFileSync, writeFileSync } from "node:fs";

import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { createRng } from "../../src/core/random/index.ts";

interface Payload {
    name: string;
    vocabulary: number;
    trainTokens: number[];
    testTokens: number[];
    externalTokens: number[];
    bytes: { test: number; external: number };
}

const names = ["bpe-2048", "bpe-2048-suffix"];
const context = 64;
const steps = 300;
const results = [];
for (const name of names) {
    const payload = JSON.parse(readFileSync(`data/gpu/${name}.json`, "utf8")) as Payload;
    const model = new TransformerLM(payload.vocabulary, {
        modelSize: 32,
        heads: 4,
        blocks: 1,
        hiddenSize: 64,
        seed: 1,
    });
    const optimizer = new SGD(model.parameters(), 0.005);
    const rng = createRng(1);
    for (let step = 0; step < steps; step++) {
        const start = rng.int(payload.trainTokens.length - context - 1);
        optimizer.zeroGrad();
        const loss = model.loss(payload.trainTokens.slice(start, start + context + 1));
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }
    const measure = (tokens: number[], byteCount: number): number => {
        const stride = Math.max(1, Math.floor((tokens.length - context - 1) / 4));
        let total = 0;
        for (let window = 0; window < 4; window++) {
            total += model.loss(tokens.slice(window * stride, window * stride + context + 1)).get();
        }
        return ((total / 4) * tokens.length) / byteCount / Math.LN2;
    };
    const row = {
        name,
        vocabulary: payload.vocabulary,
        bpbTest: measure(payload.testTokens, payload.bytes.test),
        bpbExternal: measure(payload.externalTokens, payload.bytes.external),
        parameters: model.parameterCount(),
    };
    results.push(row);
    console.log(
        `${name.padEnd(16)} vocab ${payload.vocabulary} · BPB test ${row.bpbTest.toFixed(3)} · externo ${row.bpbExternal.toFixed(3)}`,
    );
}
writeFileSync("data/tokenizer-hybrid-quick.json", `${JSON.stringify({ context, steps, results }, null, 2)}\n`, "utf8");
console.log("guardado: data/tokenizer-hybrid-quick.json");
