// Teoría: docs/teoria/evolucion/haea.md
// Experimento: minimiza la esfera 2D con HAEA y muestra la curva del mejor fitness.

import {
    ArithmeticCrossover,
    GaussianMutation,
    haea,
    UniformCrossover,
} from "../../src/evolution/index.ts";

const sphere = (genome: Float32Array): number => {
    let total = 0;
    for (const value of genome) total += value * value;
    return total;
};

const result = haea(sphere, 2, {
    populationSize: 30,
    generations: 100,
    operators: [
        new GaussianMutation({ sigma: 0.5 }),
        new UniformCrossover(),
        new ArithmeticCrossover(),
    ],
    seed: 2,
    elitism: 1,
});

console.log(
    `mejor fitness ${result.best.fitness.toFixed(6)} · genoma ` +
        `[${[...result.best.genome].map((value) => value.toFixed(4)).join(", ")}]`,
);
console.log(`evaluaciones ${result.evaluations}`);
for (let generation = 0; generation < result.history.length; generation += 10) {
    console.log(`generación ${String(generation).padStart(3)}  ${(result.history[generation] as number).toFixed(6)}`);
}
