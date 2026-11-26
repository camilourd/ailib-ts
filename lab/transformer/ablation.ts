// Teoría: docs/teoria/secuencias/ablacion-transformer.md
// Ablación: mismo LM y presupuesto, quitando RoPE, atención o FFN.

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

const variants = [
    { name: "completo", options: {} },
    { name: "sin RoPE", options: { rope: false } },
    { name: "sin atención", options: { attention: false } },
    { name: "sin FFN", options: { ffn: false } },
];

interface Row {
    name: string;
    parameters: number;
    loss: number;
    bpc: number;
    sample: string;
}

const rows: Row[] = [];
for (const variant of variants) {
    const model = new TransformerLM(256, {
        modelSize: 32,
        heads: 4,
        blocks: 1,
        hiddenSize: 64,
        seed: 1,
        ...variant.options,
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
    }
    const testLoss = model.loss(testTokens).get();
    const bpc = (testLoss * (testTokens.length - 1)) / testCharacters / Math.LN2;
    const parameters = model.parameters().reduce((total, parameter) => total + parameter.size, 0);
    const start = bytes.encode("En un ");
    const generated = model.generate(start, 50, { temperature: 0.8, seed: 7 });
    rows.push({
        name: variant.name,
        parameters,
        loss: lastLoss,
        bpc,
        sample: bytes.decode([...start, ...generated]),
    });
}

console.log("variante      parámetros  pérdida train  BPC test");
for (const row of rows) {
    console.log(
        `${row.name.padEnd(13)} ${String(row.parameters).padStart(7)}     ` +
            `${row.loss.toFixed(4)}         ${row.bpc.toFixed(3)}`,
    );
}
for (const row of rows) {
    console.log(`\n${row.name}: ${row.sample}`);
}
