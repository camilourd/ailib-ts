// Teoría: docs/teoria/nucleo/rng.md
// Experimento: histograma de 100 000 tiradas y reproducibilidad con la misma semilla.

import { createRng } from "../../src/core/random/index.ts";

const DRAWS = 100_000;
const BUCKETS = 10;
const expected = DRAWS / BUCKETS;

const rng = createRng(2026);
const counts = Array.from({ length: BUCKETS }, () => 0);
for (let i = 0; i < DRAWS; i++) {
    const bucket = Math.min(BUCKETS - 1, Math.floor(rng.next() * BUCKETS));
    counts[bucket] += 1;
}

console.log(`uniformidad: ${DRAWS} tiradas en ${BUCKETS} cubos (esperado ${expected} por cubo)`);
counts.forEach((count, index) => {
    const from = (index / BUCKETS).toFixed(1);
    const to = ((index + 1) / BUCKETS).toFixed(1);
    console.log(`  [${from}, ${to})  ${String(count).padStart(6)}  ${(count - expected).toFixed(0)}`);
});

const a = createRng(1);
const b = createRng(1);
const same = Array.from({ length: 5 }, () => (a.next() === b.next() ? "ok" : "distinto"));
console.log(`reproducibilidad (misma semilla): ${same.join(" ")}`);
