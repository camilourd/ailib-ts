import assert from "node:assert/strict";
import { test } from "node:test";

import { MLP } from "../../src/nn/mlp.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { benchmark } from "../../src/sets/runner.ts";
import { syntheticAssetSet } from "../../src/sets/tabular/synthetic-asset.ts";

test("el benchmark entrena y evalúa cada candidato", () => {
    const set = syntheticAssetSet({ days: 150, seed: 5 });
    const linear = new Linear(3, 1, { seed: 1 });
    const mlp = new MLP([3, 4, 1], { seed: 2 });
    const results = benchmark(set, [
        {
            name: "lineal",
            model: linear,
            optimizer: new SGD(linear.parameters(), 0.1),
            steps: 50,
        },
        {
            name: "mlp",
            model: mlp,
            optimizer: new SGD(mlp.parameters(), 0.1),
            steps: 50,
        },
    ]);
    assert.equal(results.length, 2);
    for (const result of results) {
        assert.equal(result.metric, set.metricName);
        assert.ok(Number.isFinite(result.value), `valor ${result.value}`);
        assert.equal(result.baseline, set.baseline);
    }
});
