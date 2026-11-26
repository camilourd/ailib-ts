// Teoría: docs/teoria/secuencias/evolucion-lm.md
// Híbrido: HAEA con un operador de 1 paso de SGD y un presupuesto global de 20 pasos.

import {
    ArithmeticCrossover,
    GaussianMutation,
    haea,
    UniformCrossover,
} from "../../src/evolution/index.ts";
import type { Genome, Variation } from "../../src/evolution/index.ts";
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
const cells: CellType[] = ["rnn", "gru", "lstm"];
const haeaGenerations = 50;
const sgdSteps = 4000;
const gradientBudget = 20;

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

function load(model: CharLM, genome: Float32Array, requiresGrad: boolean): void {
    let offset = 0;
    for (const parameter of model.parameters()) {
        parameter.data.set(genome.subarray(offset, offset + parameter.size));
        parameter.requiresGrad = requiresGrad;
        offset += parameter.size;
    }
}

function makeModel(cell: CellType): CharLM {
    return new CharLM(256, { embeddingSize: 8, hiddenSize: 16, seed: 1, cell });
}

class GradientStep implements Variation {
    readonly name = "paso-sgd";
    readonly arity = 1;
    readonly #learningRate: number;
    readonly #windows: number[][];
    readonly #makeModel: () => CharLM;
    #remaining: number;

    constructor(
        learningRate: number,
        windows: number[][],
        makeModel: () => CharLM,
        budget: number,
    ) {
        this.#learningRate = learningRate;
        this.#windows = windows;
        this.#makeModel = makeModel;
        this.#remaining = budget;
    }

    get remaining(): number {
        return this.#remaining;
    }

    apply(parents: readonly Genome[], rng: { next(): number }): Genome {
        const parent = parents[0] as Genome;
        if (this.#remaining <= 0) return new Float32Array(parent);
        this.#remaining -= 1;
        const model = this.#makeModel();
        load(model, parent, true);
        const optimizer = new SGD(model.parameters(), this.#learningRate);
        optimizer.zeroGrad();
        const piece = this.#windows[Math.floor(rng.next() * this.#windows.length)] as number[];
        const loss = model.loss(piece);
        loss.backward();
        optimizer.step();
        return flatten(model);
    }
}

interface Result {
    cell: CellType;
    method: string;
    loss: number;
    bpc: number;
    sample: string;
}

function sample(model: CharLM): string {
    const start = bytes.encode("En un ");
    return bytes.decode([...start, ...model.generate(start, 50, { temperature: 0.8, seed: 7 })]);
}

function evaluate(model: CharLM): { loss: number; bpc: number } {
    const loss = model.loss(testTokens).get();
    return { loss, bpc: (loss * (testTokens.length - 1)) / testCharacters / Math.LN2 };
}

function runHaea(cell: CellType, withGradient: boolean): Result {
    const scratch = makeModel(cell);
    for (const parameter of scratch.parameters()) parameter.requiresGrad = false;
    const genomeSize = flatten(scratch).length;
    const gradient = withGradient
        ? new GradientStep(0.1, windows, () => makeModel(cell), gradientBudget)
        : null;
    const operators: Variation[] = [
        new GaussianMutation({ sigma: 0.05 }),
        new UniformCrossover(),
        new ArithmeticCrossover(),
    ];
    if (gradient) operators.push(gradient);

    const result = haea(
        (genome) => {
            load(scratch, genome, false);
            let total = 0;
            for (const piece of windows) total += scratch.loss(piece).get();
            return total / windows.length;
        },
        genomeSize,
        {
            populationSize: 20,
            generations: haeaGenerations,
            operators,
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
    load(scratch, result.best.genome, false);
    const { loss, bpc } = evaluate(scratch);
    return {
        cell,
        method: withGradient ? "HAEA+20" : "HAEA",
        loss,
        bpc,
        sample: sample(scratch),
    };
}

function runSgd(cell: CellType): Result {
    const model = makeModel(cell);
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
    results.push(runHaea(cell, false));
    results.push(runHaea(cell, true));
    results.push(runSgd(cell));
}

console.log(
    `LM embedding 8 · oculto 16 · HAEA 1000 evaluaciones × 4 ventanas · ` +
        `híbrido con ${gradientBudget} pasos de SGD · SGD ${sgdSteps} pasos`,
);
console.log("celda  método   pérdida  BPC");
for (const result of results) {
    console.log(
        `${result.cell.padEnd(6)} ${result.method.padEnd(7)} ${result.loss.toFixed(4)}   ${result.bpc.toFixed(3)}`,
    );
}
for (const result of results.filter((item) => item.method === "HAEA+20")) {
    console.log(`\n${result.cell} · ${result.method}: ${result.sample}`);
}
