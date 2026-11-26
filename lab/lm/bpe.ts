// Teoría: docs/teoria/nucleo/bpe.md
// Experimento: BPE frente a bytes en el mismo LM y el mismo presupuesto de pasos.

import { BPETokenizer } from "../../src/data/bpe.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharLM } from "../../src/models/char-lm.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { TEXTO } from "./texto.ts";

const split = Math.floor(TEXTO.length * 0.8);
const trainText = TEXTO.slice(0, split);
const testText = TEXTO.slice(split);
const bytes = new ByteTokenizer();
const bpe = new BPETokenizer();
bpe.train(trainText, { vocabularySize: 512, minFrequency: 2 });

const byteTrain = bytes.encode(trainText);
const byteTest = bytes.encode(testText);
const bpeTrain = bpe.encode(trainText);
const bpeTest = bpe.encode(testText);

const steps = 3000;
const window = 33;

const trainModel = (model: CharLM, tokens: number[]): number => {
    const optimizer = new SGD(model.parameters(), 0.05);
    let lastLoss = 0;
    for (let step = 0; step < steps; step++) {
        const start = (step * 32) % (tokens.length - window);
        const piece = tokens.slice(start, start + window);
        optimizer.zeroGrad();
        const loss = model.loss(piece);
        loss.backward();
        optimizer.step();
        lastLoss = loss.get();
    }
    return lastLoss;
};

interface Evaluation {
    loss: number;
    bpc: number;
    tokens: number;
    characters: number;
}

const evaluate = (
    model: CharLM,
    tokens: number[],
    decode: (ids: readonly number[]) => string,
): Evaluation => {
    const sequence = tokens.slice(0, Math.min(tokens.length, 160));
    const loss = model.loss(sequence);
    const text = decode(sequence);
    const characters = [...text].length;
    const totalNats = loss.get() * (sequence.length - 1);
    return {
        loss: loss.get(),
        bpc: totalNats / characters / Math.LN2,
        tokens: sequence.length,
        characters,
    };
};

const byteModel = new CharLM(256, { embeddingSize: 32, hiddenSize: 64, seed: 1 });
const bpeModel = new CharLM(bpe.vocabularySize, { embeddingSize: 32, hiddenSize: 64, seed: 1 });
const byteLoss = trainModel(byteModel, byteTrain);
const bpeLoss = trainModel(bpeModel, bpeTrain);

const byteResult = evaluate(byteModel, byteTest, (ids) => bytes.decode(ids));
const bpeResult = evaluate(bpeModel, bpeTest, (ids) => bpe.decode(ids));
const byteCharacters = [...trainText].length;
const bpeCharacters = [...trainText].length;

console.log(
    `texto: ${TEXTO.length} caracteres · train ${trainText.length} / test ${testText.length}`,
);
console.log(`BPE: vocabulario ${bpe.vocabularySize} · ${bpe.merges.length} merges`);
console.log("");
console.log("tokenizer  vocab  tokens/char  pérdida/token  BPC test  tokens test");
console.log(
    `bytes      ${String(256).padEnd(6)} ${(byteTrain.length / byteCharacters).toFixed(3)}        ` +
        `${byteLoss.toFixed(4)}          ${byteResult.bpc.toFixed(3)}     ${byteResult.tokens}`,
);
console.log(
    `bpe        ${String(bpe.vocabularySize).padEnd(6)} ${(bpeTrain.length / bpeCharacters).toFixed(3)}        ` +
        `${bpeLoss.toFixed(4)}          ${bpeResult.bpc.toFixed(3)}     ${bpeResult.tokens}`,
);

const byteStart = bytes.encode("En un ");
const bpeStart = bpe.encode("En un ");
console.log("\n--- generado con bytes ---");
console.log(bytes.decode([...byteStart, ...byteModel.generate(byteStart, 80, { temperature: 0.8, seed: 7 })]));
console.log("\n--- generado con BPE ---");
console.log(bpe.decode([...bpeStart, ...bpeModel.generate(bpeStart, 80, { temperature: 0.8, seed: 7 })]));
