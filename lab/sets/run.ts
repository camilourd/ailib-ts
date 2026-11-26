// Teoría: docs/teoria/entrenamiento/sets.md
// Experimento: compara una regresión lineal y un MLP sobre el activo sintético.

import { MLP } from "../../src/nn/mlp.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { benchmark } from "../../src/sets/runner.ts";
import { syntheticAssetSet } from "../../src/sets/tabular/synthetic-asset.ts";

const set = syntheticAssetSet({ days: 800, seed: 7 });
const linear = new Linear(3, 1, { seed: 1 });
const mlp = new MLP([3, 8, 1], { seed: 2 });

const results = benchmark(set, [
    { name: "lineal", model: linear, optimizer: new SGD(linear.parameters(), 0.1), steps: 500 },
    { name: "mlp", model: mlp, optimizer: new SGD(mlp.parameters(), 0.1), steps: 500 },
]);

console.log(`set: ${set.name} · ${set.description}`);
console.log(`baseline (${set.metricName}): ${set.baseline.toFixed(4)}`);
for (const result of results) {
    console.log(`${result.name}: ${result.metric} ${result.value.toFixed(4)}`);
}
