import assert from "node:assert/strict";
import { test } from "node:test";

import { createRng, Mulberry32, Uint32RandomGenerator } from "../../../src/core/random/index.ts";

const SEED_1 = [
    0.62707394058816135, 0.0027357211802154779, 0.52744703995995224, 0.98105096747167408,
    0.96837789821438491,
];

/** Generador de prueba: devuelve una secuencia fija de uint32 en ciclo. */
class FixedSequence extends Uint32RandomGenerator {
    #values: number[];
    #index = 0;

    constructor(values: number[]) {
        super();
        this.#values = values;
    }

    protected nextUint32(): number {
        const value = this.#values[this.#index % this.#values.length] as number;
        this.#index += 1;
        return value;
    }
}

test("la base deriva next() del paso crudo", () => {
    const rng = new FixedSequence([0x00000000, 0x40000000, 0x80000000, 0xc0000000]);
    assert.equal(rng.next(), 0);
    assert.equal(rng.next(), 0.25);
    assert.equal(rng.next(), 0.5);
    assert.equal(rng.next(), 0.75);
});

test("la base deriva int() con rechazo y shuffle() como permutación", () => {
    const rng = new FixedSequence([0, 1, 2, 3]);
    assert.deepEqual([rng.int(2), rng.int(2), rng.int(2), rng.int(2)], [0, 1, 0, 1]);
    const shuffled = new FixedSequence([0, 1, 2]).shuffle([10, 20, 30]);
    assert.deepEqual([...shuffled].sort((a, b) => a - b), [10, 20, 30]);
});

test("misma semilla, misma secuencia", () => {
    const a = new Mulberry32(42);
    const b = new Mulberry32(42);
    for (let i = 0; i < 100; i++) {
        assert.equal(a.next(), b.next(), `difiere en el paso ${i}`);
    }
});

test("createRng devuelve un generador equivalente a Mulberry32", () => {
    const a = createRng(7);
    const b = new Mulberry32(7);
    for (let i = 0; i < 10; i++) {
        assert.equal(a.next(), b.next());
    }
});

test("semillas distintas dan secuencias distintas", () => {
    const a = createRng(1);
    const b = createRng(2);
    let equal = 0;
    for (let i = 0; i < 20; i++) {
        if (a.next() === b.next()) equal++;
    }
    assert.equal(equal, 0);
});

test("next() está en [0, 1)", () => {
    const rng = createRng(7);
    for (let i = 0; i < 10_000; i++) {
        const value = rng.next();
        assert.ok(value >= 0 && value < 1, `fuera de rango: ${value}`);
    }
});

test("int() respeta el rango y se reparte por cubos", () => {
    const rng = createRng(3);
    const counts = Array.from({ length: 6 }, () => 0);
    const draws = 60_000;
    for (let i = 0; i < draws; i++) {
        counts[rng.int(6)] += 1;
    }
    for (const count of counts) {
        assert.ok(Math.abs(count - draws / 6) < 500, `cubo desviado: ${count}`);
    }
});

test("int() rechaza límites inválidos", () => {
    const rng = createRng(1);
    assert.throws(() => rng.int(0), RangeError);
    assert.throws(() => rng.int(-3), RangeError);
    assert.throws(() => rng.int(2.5), RangeError);
    assert.throws(() => rng.int(0x100000001), RangeError);
});

test("shuffle es una permutación y no muta la entrada", () => {
    const rng = createRng(9);
    const input = Array.from({ length: 20 }, (_, index) => index);
    const output = rng.shuffle(input);
    assert.deepEqual([...output].sort((a, b) => a - b), input);
    assert.deepEqual(input, Array.from({ length: 20 }, (_, index) => index));
    assert.deepEqual(createRng(9).shuffle(input), output);
});

test("regresión: la semilla 1 produce la secuencia esperada", () => {
    const rng = createRng(1);
    SEED_1.forEach((expected, index) => {
        assert.ok(Math.abs(rng.next() - expected) < 1e-15, `difiere en el paso ${index}`);
    });
});
