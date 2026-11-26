// Teoría: docs/teoria/secuencias/evolucion-lm.md
// Experimento: entrenar RNN, GRU y LSTM con HAEA frente a SGD en el mismo LM pequeño.

import { haea, ArithmeticCrossover, GaussianMutation, UniformCrossover } from "../../src/evolution/index.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharLM } from "../../src/models/char-lm.ts";
import type { CellType } from "../../src/nn/recurrent.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { TEXTO } from "./texto.ts";

const bytes = new ByteTokenizer();
const split = Math.floor(TEXTO.length * 0.8);
const trainTokens = bytes.encode(TEXTO.slice(0, split));
const testTokens = bytes.encode(TEXTO.slice(split)).slice(0, 64);
const testCharacters = [...bytes.decode(testTokens)].length;
const window = 33;
const windows = [0, 150, 300, 450].map((start) => trainTokens.slice(start, start + window));
const haeaGenerations = 50;
const sgdSteps = 4000;
const cells: CellType[] = ["rnn", "gru", "lstm"];

function flatten(model: CharLM): Float32Array {
    const parameters = model.parameters();
    const size = parameters.reduce((total, parameter) => total + parameter.size, 0);
    const genome = new Float32Array(size);
    let offset = 0;
    for (const parameter of parameters) {
        genome.set(parameter.data, offset);
        offset += parameter.size;
    }
    return genome;
}

function setGenome(model: CharLM, genome: Float32Array): void {
    let offset = 0;
    for (const parameter of model.parameters()) {
        parameter.data.set(genome.subarray(offset, offset + parameter.size));
        parameter.requiresGrad = false;
        offset += parameter.size;
    }
}

interface Result {
    cell: CellType;
    method: string;
    loss: number;
    bpc: number;
    sample: string;
}

function evaluate(model: CharLM): { loss: number; bpc: number } {
    const loss = model.loss(testTokens).get();
    return {
        loss,
        bpc: (loss * (testTokens.length - 1)) / testCharacters / Math.LN2,
    };
}

function sample(model: CharLM): string {
    const start = bytes.encode("En un ");
    return bytes.decode([...start, ...model.generate(start, 50, { temperature: 0.8, seed: 7 })]);
}

function runHaea(cell: CellType): Result {
    const model = new CharLM(256, { embeddingSize: 8, hiddenSize: 16, seed: 1, cell });
    for (const parameter of model.parameters()) parameter.requiresGrad = false;
    const genomeSize = flatten(model).length;
    const result = haea(
        (genome) => {
            setGenome(model, genome);
            let total = 0;
            for (const piece of windows) total += model.loss(piece).get();
            return total / windows.length;
        },
        genomeSize,
        {
            populationSize: 20,
            generations: haeaGenerations,
            operators: [
                new GaussianMutation({ sigma: 0.05 }),
                new UniformCrossover(),
                new ArithmeticCrossover(),
            ],
            seed: 1,
            elitism: 1,
            tournamentSize: 3,
            init: (rng) => {
                const genome = new Float32Array(genomeSize);
                for (let index = 0; index < genome.length; index++) genome[index] = rng.next() * 0.2 - 0.1;
                return genome;
            },
        },
    );
    setGenome(model, result.best.genome);
    const { loss, bpc } = evaluate(model);
    return { cell, method: "HAEA", loss, bpc, sample: sample(model) };
}

function runSgd(cell: CellType): Result {
    const model = new CharLM(256, { embeddingSize: 8, hiddenSize: 16, seed: 1, cell });
    const optimizer = new SGD(model.parameters(), 0.1);
    for (let step = 0; step < sgdSteps; step++) {
        const start = (step * 32) % (trainTokens.length - window);
        const piece = trainTokens.slice(start, start + window);
        optimizer.zeroGrad();
        const loss = model.loss(piece);
        loss.backward();
        optimizer.step();
    }
    const { loss, bpc } = evaluate(model);
    return { cell, method: "SGD", loss, bpc, sample: sample(model) };
}

const results: Result[] = [];
for (const cell of cells) {
    results.push(runHaea(cell));
    results.push(runSgd(cell));
}

console.log(
    `LM pequeño: embedding 8, oculto 16 · HAEA ${haeaGenerations * 20} evaluaciones × 4 ventanas · ` +
        `SGD ${sgdSteps} pasos × 1 ventana · train ${trainTokens.length} bytes · test ${testTokens.length} bytes`,
);
console.log("celda  método  pérdida test  BPC test");
for (const result of results) {
    console.log(
        `${result.cell.padEnd(6)} ${result.method.padEnd(6)} ${result.loss.toFixed(4)}       ${result.bpc.toFixed(3)}`,
    );
}
for (const result of results) {
    console.log(`\n${result.cell} · ${result.method}: ${result.sample}`);
}
