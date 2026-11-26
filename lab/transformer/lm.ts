// Teoría: docs/teoria/secuencias/lm-transformer.md
// Experimento: LM transformer pequeño con RoPE, máscara causal y un bloque.

import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { TEXTO } from "../lm/texto.ts";

const bytes = new ByteTokenizer();
const tokens = bytes.encode(TEXTO);
const split = Math.floor(tokens.length * 0.8);
const trainTokens = tokens.slice(0, split);
const testTokens = tokens.slice(split, split + 64);
const testCharacters = [...bytes.decode(testTokens)].length;
const context = 33;
const steps = 3000;

const model = new TransformerLM(256, {
    modelSize: 32,
    heads: 4,
    blocks: 1,
    hiddenSize: 64,
    seed: 1,
});
const optimizer = new SGD(model.parameters(), 0.01);

let lastLoss = 0;
for (let step = 0; step < steps; step++) {
    const start = (step * 32) % (trainTokens.length - context);
    const piece = trainTokens.slice(start, start + context);
    optimizer.zeroGrad();
    const loss = model.loss(piece);
    loss.backward();
    optimizer.step();
    lastLoss = loss.get();
    if (step % 500 === 0) {
        console.log(`paso ${String(step).padStart(4)}  pérdida ${lastLoss.toFixed(4)}`);
    }
}

const testLoss = model.loss(testTokens).get();
const bpc = (testLoss * (testTokens.length - 1)) / testCharacters / Math.LN2;
console.log(`pérdida final ${lastLoss.toFixed(4)} · BPC test ${bpc.toFixed(3)}`);
const start = bytes.encode("En un ");
const generated = model.generate(start, 80, { temperature: 0.8, seed: 7 });
console.log("--- generado ---");
console.log(bytes.decode([...start, ...generated]));
