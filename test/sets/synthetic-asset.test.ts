import assert from "node:assert/strict";
import { test } from "node:test";

import { syntheticAssetSet } from "../../src/sets/tabular/synthetic-asset.ts";

test("el set sintético tiene formas y splits cronológicos", () => {
    const set = syntheticAssetSet({ days: 200, seed: 3 });
    assert.equal(set.kind, "regression");
    assert.equal(set.metricName, "MAE");
    const sizes = [
        set.train.input.shape[0],
        set.validation.input.shape[0],
        set.test.input.shape[0],
    ];
    assert.deepEqual(sizes, [132, 28, 29]);
    for (const split of [set.train, set.validation, set.test]) {
        assert.deepEqual(split.input.shape, [split.input.shape[0], 3]);
        assert.deepEqual(split.target.shape, [split.target.shape[0], 1]);
    }
    assert.ok(set.baseline > 0);
});

test("la normalización usa solo el train", () => {
    const set = syntheticAssetSet({ days: 200, seed: 3 });
    const rows = set.train.input.shape[0];
    for (let feature = 0; feature < 3; feature++) {
        let sum = 0;
        let squares = 0;
        for (let row = 0; row < rows; row++) {
            const value = set.train.input.get(row, feature);
            sum += value;
            squares += value * value;
        }
        const mean = sum / rows;
        const deviation = Math.sqrt(squares / rows - mean * mean);
        assert.ok(Math.abs(mean) < 1e-5, `media ${mean}`);
        assert.ok(Math.abs(deviation - 1) < 1e-4, `desviación ${deviation}`);
    }
});

test("la misma semilla da los mismos datos y otra semilla cambia", () => {
    const first = syntheticAssetSet({ days: 200, seed: 3 });
    const second = syntheticAssetSet({ days: 200, seed: 3 });
    const other = syntheticAssetSet({ days: 200, seed: 4 });
    assert.deepEqual([...first.train.input.data], [...second.train.input.data]);
    assert.notDeepEqual([...first.train.input.data], [...other.train.input.data]);
});
