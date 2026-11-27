// Teoría: docs/teoria/nucleo/disjoint-set-union.md
// Experimento: unir componentes y mostrar los grupos.

import { DisjointSetUnion } from "../../src/core/disjoint-set-union.ts";

const dsu = new DisjointSetUnion(8);
for (const [first, second] of [
    [0, 1],
    [1, 2],
    [3, 4],
    [5, 6],
    [4, 2],
] as const) {
    dsu.join(first, second);
}

const groups = new Map<number, number[]>();
for (let value = 0; value < dsu.size; value++) {
    const root = dsu.find(value);
    const group = groups.get(root) ?? [];
    group.push(value);
    groups.set(root, group);
}
console.log([...groups.values()].map((group) => group.join("-")).join(" · "));
