// Teoría: docs/teoria/evolucion/evolucion-vs-gradiente.md
// Experimento: HAEA (adaptativo y fijo) contra SGD en XOR, con el mismo presupuesto de ejemplos.

import { Tensor } from "../../src/core/tensor.ts";
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
const POPULATION = 30;
const GENERATIONS = 200;
const STEPS = POPULATION * (GENERATIONS + 1);
const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const EXAMPLES_PER_INDIVIDUAL = 4;

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
    seed: number;
    finalLoss: number;
    accuracy: number;
    examples: number | null;
    success: boolean;
}

function runHaea(seed: number, adaptive: boolean): Outcome {
    const result = haea(fitness, 22, {
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
            const genome = new Float32Array(22);
            for (let index = 0; index < genome.length; index++) {
                genome[index] = rng.next() - 0.5;
            }
            return genome;
        },
    });
    setGenome(scratch, result.best.genome);
    const accuracy = accuracyOf(scratch);
    const reached = result.history.findIndex((value) => value < 0.1);
    return {
        method: adaptive ? "HAEA adaptativo" : "HAEA fijo",
        seed,
        finalLoss: result.best.fitness,
        accuracy,
        examples: reached < 0 ? null : (reached + 1) * POPULATION * EXAMPLES_PER_INDIVIDUAL,
        success: accuracy === 1,
    };
}

function runSgd(seed: number): Outcome {
    const model = new MLP(SIZES, { seed });
    const optimizer = new SGD(model.parameters(), 0.5);
    let firstStep: number | null = null;
    const history = train((input) => model.forward(input), INPUTS, {
        steps: STEPS,
        optimizer,
        loss: (prediction) => crossEntropy(prediction, TARGETS),
        onStep: (step, loss) => {
            if (firstStep === null && loss < 0.1) firstStep = step;
        },
    });
    return {
        method: "SGD",
        seed,
        finalLoss: history.at(-1) as number,
        accuracy: accuracyOf(model),
        examples: firstStep === null ? null : (firstStep + 1) * EXAMPLES_PER_INDIVIDUAL,
        success: accuracyOf(model) === 1,
    };
}

const outcomes = SEEDS.flatMap((seed) => [runSgd(seed), runHaea(seed, true), runHaea(seed, false)]);
const methods = ["SGD", "HAEA adaptativo", "HAEA fijo"];

const mean = (values: number[]): number =>
    values.length === 0 ? Number.NaN : values.reduce((total, value) => total + value, 0) / values.length;

console.log(
    `XOR · ${POPULATION} individuos × ${GENERATIONS + 1} evaluaciones = ${STEPS * EXAMPLES_PER_INDIVIDUAL} ejemplos`,
);
console.log("método            éxito   CE<0.1   ejemplos hasta CE<0.1   CE final   exactitud");
for (const method of methods) {
    const group = outcomes.filter((outcome) => outcome.method === method);
    const successes = group.filter((outcome) => outcome.success);
    const reached = successes
        .map((outcome) => outcome.examples)
        .filter((value): value is number => value !== null);
    console.log(
        `${method.padEnd(17)} ${successes.length}/${group.length}    ` +
            `${reached.length}/${group.length}      ` +
            `${reached.length > 0 ? mean(reached).toFixed(0) : "—"}                    ` +
            `${mean(group.map((outcome) => outcome.finalLoss)).toFixed(4)}     ` +
            `${mean(group.map((outcome) => outcome.accuracy)).toFixed(2)}`,
    );
}
