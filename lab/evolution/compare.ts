// Teoría: docs/teoria/evolucion/evolucion-vs-gradiente.md
// Comparación completa: azar, SGD y HAEA (fijo y adaptativo) en XOR, 30 semillas.

import { writeFileSync } from "node:fs";

import { Tensor } from "../../src/core/tensor.ts";
import { createRng } from "../../src/core/random/index.ts";
import {
    ArithmeticCrossover,
    GaussianMutation,
    haea,
    UniformCrossover,
} from "../../src/evolution/index.ts";
import { crossEntropy } from "../../src/nn/losses.ts";
import { MLP } from "../../src/nn/mlp.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { train } from "../../src/train/trainer.ts";

const INPUTS = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
const TARGETS = [0, 1, 1, 0];
const SIZES = [2, 4, 2];
const GENOME_SIZE = 22;
const POPULATION = 30;
const GENERATIONS = 200;
const STEPS = POPULATION * (GENERATIONS + 1);
const EXAMPLES_PER_INDIVIDUAL = 4;
const BUDGET = STEPS * EXAMPLES_PER_INDIVIDUAL;
const SEEDS = Array.from({ length: 30 }, (_, index) => index + 1);
const CHECKPOINTS = [0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000];

const scratch = new MLP(SIZES, { seed: 0 });
for (const parameter of scratch.parameters()) parameter.requiresGrad = false;

function setGenome(model: MLP, genome: Float32Array): void {
    let offset = 0;
    for (const parameter of model.parameters()) {
        parameter.data.set(genome.subarray(offset, offset + parameter.size));
        offset += parameter.size;
    }
}

function fitness(genome: Float32Array): number {
    setGenome(scratch, genome);
    return crossEntropy(scratch.forward(INPUTS), TARGETS).get();
}

function accuracyOf(model: MLP): number {
    const logits = model.forward(INPUTS);
    let correct = 0;
    for (let row = 0; row < 4; row++) {
        const predicted = logits.get(row, 1) > logits.get(row, 0) ? 1 : 0;
        if (predicted === TARGETS[row]) correct += 1;
    }
    return correct / 4;
}

interface Outcome {
    method: string;
    finalLoss: number;
    accuracy: number;
    success: boolean;
    reached: boolean;
    examples: number;
    history: number[];
    examplesPerPoint: number;
    seconds: number;
}

function runHaea(seed: number, adaptive: boolean): Outcome {
    const start = performance.now();
    const result = haea(fitness, GENOME_SIZE, {
        populationSize: POPULATION,
        generations: GENERATIONS,
        operators: [
            new GaussianMutation({ sigma: 0.5 }),
            new UniformCrossover(),
            new ArithmeticCrossover(),
        ],
        seed,
        adaptive,
        elitism: 1,
        tournamentSize: 3,
        init: (rng) => {
            const genome = new Float32Array(GENOME_SIZE);
            for (let index = 0; index < genome.length; index++) genome[index] = rng.next() - 0.5;
            return genome;
        },
    });
    setGenome(scratch, result.best.genome);
    const accuracy = accuracyOf(scratch);
    const reached = result.history.findIndex((value) => value < 0.1);
    return {
        method: adaptive ? "HAEA adaptativo" : "HAEA fijo",
        finalLoss: result.best.fitness,
        accuracy,
        success: accuracy === 1,
        reached: reached >= 0,
        examples: reached < 0 ? BUDGET : (reached + 1) * POPULATION * EXAMPLES_PER_INDIVIDUAL,
        history: result.history,
        examplesPerPoint: POPULATION * EXAMPLES_PER_INDIVIDUAL,
        seconds: (performance.now() - start) / 1000,
    };
}

function runSgd(seed: number): Outcome {
    const start = performance.now();
    const model = new MLP(SIZES, { seed });
    const optimizer = new SGD(model.parameters(), 0.5);
    const history: number[] = [];
    let firstStep: number | null = null;
    train((input) => model.forward(input), INPUTS, {
        steps: STEPS,
        optimizer,
        loss: (prediction) => crossEntropy(prediction, TARGETS),
        onStep: (step, loss) => {
            history.push(loss);
            if (firstStep === null && loss < 0.1) firstStep = step;
        },
    });
    const accuracy = accuracyOf(model);
    return {
        method: "SGD",
        finalLoss: crossEntropy(model.forward(INPUTS), TARGETS).get(),
        accuracy,
        success: accuracy === 1,
        reached: firstStep !== null,
        examples: firstStep === null ? BUDGET : (firstStep + 1) * EXAMPLES_PER_INDIVIDUAL,
        history,
        examplesPerPoint: EXAMPLES_PER_INDIVIDUAL,
        seconds: (performance.now() - start) / 1000,
    };
}

function runRandom(seed: number): Outcome {
    const start = performance.now();
    const rng = createRng(seed);
    const history: number[] = [];
    let best = Number.POSITIVE_INFINITY;
    let bestGenome = new Float32Array(GENOME_SIZE);
    let reachedAt: number | null = null;
    for (let index = 0; index < STEPS; index++) {
        const genome = new Float32Array(GENOME_SIZE);
        for (let gene = 0; gene < GENOME_SIZE; gene++) genome[gene] = rng.next() - 0.5;
        const loss = fitness(genome);
        if (loss < best) {
            best = loss;
            bestGenome = genome;
        }
        history.push(best);
        if (reachedAt === null && best < 0.1) reachedAt = index + 1;
    }
    setGenome(scratch, bestGenome);
    const accuracy = accuracyOf(scratch);
    return {
        method: "Azar",
        finalLoss: best,
        accuracy,
        success: accuracy === 1,
        reached: reachedAt !== null,
        examples: reachedAt === null ? BUDGET : reachedAt * EXAMPLES_PER_INDIVIDUAL,
        history,
        examplesPerPoint: EXAMPLES_PER_INDIVIDUAL,
        seconds: (performance.now() - start) / 1000,
    };
}

const outcomes = SEEDS.flatMap((seed) => [
    runRandom(seed),
    runSgd(seed),
    runHaea(seed, false),
    runHaea(seed, true),
]);

const methods = ["Azar", "SGD", "HAEA fijo", "HAEA adaptativo"];
const mean = (values: number[]): number =>
    values.reduce((total, value) => total + value, 0) / values.length;
const deviation = (values: number[]): number => {
    const average = mean(values);
    return Math.sqrt(mean(values.map((value) => (value - average) ** 2)));
};
const median = (values: number[]): number => {
    const sorted = [...values].sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
        : (sorted[middle] as number);
};

const sample = (outcome: Outcome, budget: number): number => {
    const index = Math.min(
        outcome.history.length - 1,
        Math.max(0, Math.ceil(budget / outcome.examplesPerPoint) - 1),
    );
    return outcome.history[index] as number;
};

console.log(
    `XOR · ${SEEDS.length} semillas · ${BUDGET} ejemplos por método · ` +
        `${POPULATION} individuos × ${GENERATIONS + 1} generaciones`,
);
console.log(
    "método            éxito   CE<0.1   ejemplos (mediana)   CE final (media±sd)   exactitud   tiempo",
);
for (const method of methods) {
    const group = outcomes.filter((outcome) => outcome.method === method);
    const successes = group.filter((outcome) => outcome.success).length;
    const reached = group.filter((outcome) => outcome.reached).length;
    const losses = group.map((outcome) => outcome.finalLoss);
    console.log(
        `${method.padEnd(17)} ${successes}/${group.length}    ${reached}/${group.length}      ` +
            `${median(group.map((outcome) => outcome.examples)).toFixed(0).padStart(6)}             ` +
            `${mean(losses).toFixed(4)} ± ${deviation(losses).toFixed(4)}      ` +
            `${mean(group.map((outcome) => outcome.accuracy)).toFixed(2)}        ` +
            `${mean(group.map((outcome) => outcome.seconds)).toFixed(2)}s`,
    );
}

const colors: Record<string, string> = {
    Azar: "#888888",
    SGD: "#1f5fa8",
    "HAEA fijo": "#2f8f46",
    "HAEA adaptativo": "#c26a1b",
};
const width = 880;
const height = 460;
const left = 70;
const right = 30;
const top = 50;
const bottom = 60;
const x = (budget: number): number => left + (budget / BUDGET) * (width - left - right);
const y = (value: number): number => top + (1 - value) * (height - top - bottom);

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
    `<text x="${left}" y="28" font-family="sans-serif" font-size="16" fill="#222">XOR: proporción de semillas con CE &lt; 0.1</text>`,
];
for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(
        `<line x1="${left}" y1="${y(fraction)}" x2="${width - right}" y2="${y(fraction)}" stroke="#dddddd"/>`,
        `<text x="${left - 10}" y="${y(fraction) + 4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">${(fraction * 100).toFixed(0)}%</text>`,
    );
}
for (const budget of [0, 4000, 8000, 12000, 16000, 20000, 24000]) {
    parts.push(
        `<line x1="${x(budget)}" y1="${top}" x2="${x(budget)}" y2="${height - bottom}" stroke="#eeeeee"/>`,
        `<text x="${x(budget)}" y="${height - bottom + 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">${budget / 1000}k</text>`,
    );
}
for (const method of methods) {
    const group = outcomes.filter((outcome) => outcome.method === method);
    const points = CHECKPOINTS.map((budget) => {
        const successes = group.filter((outcome) => sample(outcome, budget) < 0.1).length;
        return `${x(budget)},${y(successes / group.length)}`;
    }).join(" ");
    parts.push(
        `<polyline points="${points}" fill="none" stroke="${colors[method]}" stroke-width="2.5"/>`,
    );
}
methods.forEach((method, index) => {
    const legendX = width - 210;
    const legendY = top + 18 + index * 18;
    parts.push(
        `<line x1="${legendX}" y1="${legendY}" x2="${legendX + 18}" y2="${legendY}" stroke="${colors[method]}" stroke-width="3"/>`,
        `<text x="${legendX + 24}" y="${legendY + 4}" font-family="sans-serif" font-size="12" fill="#333">${method}</text>`,
    );
});
parts.push(
    `<text x="${left}" y="${height - 12}" font-family="sans-serif" font-size="11" fill="#555">ejemplos evaluados</text>`,
    `</svg>`,
);

const svgPath = "docs/teoria/evolucion/comparacion.svg";
writeFileSync(svgPath, `${parts.join("\n")}\n`, "utf8");
console.log(`\ncurva: ${svgPath}`);
