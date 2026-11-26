import assert from "node:assert/strict";
import { test } from "node:test";

import { createRng } from "../../src/core/random/index.ts";
import { haea } from "../../src/evolution/haea.ts";
import {
    ArithmeticCrossover,
    GaussianMutation,
    UniformCrossover,
} from "../../src/evolution/operators.ts";
import { tournament } from "../../src/evolution/selection.ts";
import type { Individual, Variation } from "../../src/evolution/types.ts";

const sphere = (genome: Float32Array): number => {
    let total = 0;
    for (const value of genome) total += value * value;
    return total;
};

test("HAEA baja la esfera y el historial no empeora", () => {
    const result = haea(sphere, 2, {
        populationSize: 20,
        generations: 150,
        operators: [new GaussianMutation({ sigma: 0.5 })],
        seed: 1,
        elitism: 1,
    });
    assert.ok(result.best.fitness < 0.1, `fitness ${result.best.fitness}`);
    assert.equal(result.history.length, 151);
    assert.equal(result.evaluations, 20 * 151);
    for (let index = 1; index < result.history.length; index++) {
        assert.ok(
            (result.history[index] as number) <= (result.history[index - 1] as number) + 1e-6,
            `empeoró en ${index}`,
        );
    }
});

test("la misma semilla da el mismo resultado", () => {
    const options = {
        populationSize: 12,
        generations: 30,
        operators: [new GaussianMutation({ sigma: 0.4 }), new UniformCrossover()],
        seed: 9,
    };
    const first = haea(sphere, 3, options);
    const second = haea(sphere, 3, options);
    assert.deepEqual([...first.best.genome], [...second.best.genome]);
    assert.deepEqual(first.history, second.history);
});

test("mutación, cruce uniforme y cruce aritmético", () => {
    const mutation = new GaussianMutation({ sigma: 0.5 });
    const parent = new Float32Array([1, 2]);
    const child = mutation.apply([parent], createRng(1));
    assert.equal(child.length, 2);
    assert.notDeepEqual([...child], [...parent]);
    assert.deepEqual(
        [...mutation.apply([parent], createRng(1))],
        [...mutation.apply([parent], createRng(1))],
    );

    const first = new Float32Array([1, 2, 3]);
    const second = new Float32Array([4, 5, 6]);
    const uniform = new UniformCrossover();
    const mixed = uniform.apply([first, second], createRng(2));
    mixed.forEach((value, index) => {
        assert.ok(
            value === (first[index] as number) || value === (second[index] as number),
            `gen ${index} no viene de un padre`,
        );
    });

    const arithmetic = new ArithmeticCrossover();
    const averaged = arithmetic.apply([first, second], createRng(3));
    averaged.forEach((value, index) => {
        const low = Math.min(first[index] as number, second[index] as number);
        const high = Math.max(first[index] as number, second[index] as number);
        assert.ok(value >= low && value <= high, `gen ${index} fuera de rango`);
    });

    assert.throws(() => uniform.apply([first, new Float32Array([1])], createRng(4)), Error);
});

test("el torneo elige de la población y valida el tamaño", () => {
    const population: Individual[] = [
        { genome: new Float32Array([1]), fitness: 3 },
        { genome: new Float32Array([2]), fitness: 1 },
        { genome: new Float32Array([3]), fitness: 2 },
    ];
    const chosen = tournament(population, 1, createRng(5));
    assert.ok(population.includes(chosen));
    assert.throws(() => tournament([], 1, createRng(5)), Error);
    assert.throws(() => tournament(population, 0, createRng(5)), RangeError);
});

test("onGeneration recibe cada generación", () => {
    const seen: number[] = [];
    haea(sphere, 2, {
        populationSize: 6,
        generations: 3,
        operators: [new GaussianMutation({ sigma: 0.5 })],
        seed: 1,
        onGeneration: (generation) => seen.push(generation),
    });
    assert.deepEqual(seen, [0, 1, 2]);
});

test("HAEA valida genoma, población, generaciones y operadores", () => {
    const operator = new GaussianMutation();
    assert.throws(
        () => haea(sphere, 0, { populationSize: 4, generations: 1, operators: [operator] }),
        RangeError,
    );
    assert.throws(
        () => haea(sphere, 2, { populationSize: 1, generations: 1, operators: [operator] }),
        RangeError,
    );
    assert.throws(
        () => haea(sphere, 2, { populationSize: 4, generations: -1, operators: [operator] }),
        RangeError,
    );
    assert.throws(
        () => haea(sphere, 2, { populationSize: 4, generations: 1, operators: [] }),
        Error,
    );
    const badOperator = { name: "malo", arity: 0, apply: () => new Float32Array(2) } as Variation;
    assert.throws(
        () => haea(sphere, 2, { populationSize: 4, generations: 1, operators: [badOperator] }),
        Error,
    );
});
