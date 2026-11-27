import assert from "node:assert/strict";
import { test } from "node:test";

import { DisjointSetUnion } from "../../src/core/disjoint-set-union.ts";

test("DSU: cada elemento empieza aislado y join une conjuntos", () => {
    const dsu = new DisjointSetUnion(6);
    assert.equal(dsu.find(0), 0);
    assert.equal(dsu.find(5), 5);
    dsu.join(0, 1);
    dsu.join(1, 2);
    assert.equal(dsu.find(0), dsu.find(2));
    assert.notEqual(dsu.find(0), dsu.find(3));
});

test("DSU: join es transitivo e idempotente", () => {
    const dsu = new DisjointSetUnion(5);
    dsu.join(0, 1);
    dsu.join(3, 4);
    dsu.join(1, 3);
    dsu.join(2, 0);
    const root = dsu.find(0);
    for (const value of [1, 2, 3, 4]) {
        assert.equal(dsu.find(value), root, `valor ${value}`);
    }
    dsu.join(0, 4);
    assert.equal(dsu.find(0), root);
});

test("DSU: restart vuelve a conjuntos aislados", () => {
    const dsu = new DisjointSetUnion(3);
    dsu.join(0, 1);
    dsu.restart(4);
    assert.equal(dsu.size, 4);
    assert.equal(dsu.find(0), 0);
    assert.equal(dsu.find(1), 1);
    assert.equal(dsu.find(3), 3);
});

test("DSU: valida tamaño e índices", () => {
    assert.throws(() => new DisjointSetUnion(-1), Error);
    const dsu = new DisjointSetUnion(2);
    assert.throws(() => dsu.find(2), RangeError);
    assert.throws(() => dsu.join(-1, 0), RangeError);
});
