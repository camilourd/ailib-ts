// Teoría: docs/teoria/secuencias/lm-caracteres.md
// Experimento: entrena un LM de caracteres byte-level y genera una continuación.

import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharLM } from "../../src/models/char-lm.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { TEXTO } from "./texto.ts";

const tokenizer = new ByteTokenizer();
const tokens = tokenizer.encode(TEXTO);
const model = new CharLM(256, { embeddingSize: 32, hiddenSize: 64, seed: 1 });
const optimizer = new SGD(model.parameters(), 0.05);

const window = 33;
const steps = 10000;
let lastLoss = 0;
for (let step = 0; step < steps; step++) {
    const start = (step * 32) % (tokens.length - window);
    const piece = tokens.slice(start, start + window);
    optimizer.zeroGrad();
    const loss = model.loss(piece);
    loss.backward();
    optimizer.step();
    lastLoss = loss.get();
    if (step % 1000 === 0) {
        console.log(`paso ${String(step).padStart(4)}  pérdida ${lastLoss.toFixed(4)}`);
    }
}

console.log(`pérdida final ${lastLoss.toFixed(4)} · ${tokens.length} bytes de texto`);
const start = tokenizer.encode("En un ");
const generated = model.generate(start, 160, { temperature: 0.8, seed: 7 });
console.log("--- generado ---");
console.log(tokenizer.decode([...start, ...generated]));
