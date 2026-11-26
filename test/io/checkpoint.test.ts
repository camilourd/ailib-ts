import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import {
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    loadCheckpoint,
    saveCheckpoint,
} from "../../src/io/checkpoint.ts";

const directory = () => mkdtempSync(join(tmpdir(), "ailib-checkpoint-"));

test("guarda y carga parámetros y metadatos", (context) => {
    const dir = directory();
    context.after(() => rmSync(dir, { recursive: true, force: true }));
    const path = join(dir, "model.json");

    saveCheckpoint(
        path,
        {
            "hidden.weights": Tensor.from([1, 2, 3, 4], [2, 2]),
            "hidden.bias": Tensor.zeros([2]),
        },
        { model: "prueba", step: 3 },
    );

    const loaded = loadCheckpoint(path);
    assert.deepEqual(loaded.metadata, { model: "prueba", step: 3 });
    assert.deepEqual(loaded.parameters["hidden.weights"].shape, [2, 2]);
    assert.deepEqual([...(loaded.parameters["hidden.weights"].data ?? [])], [1, 2, 3, 4]);
    assert.deepEqual([...(loaded.parameters["hidden.bias"].data ?? [])], [0, 0]);
    assert.equal(loaded.parameters["hidden.weights"].requiresGrad, false);
});

test("loadCheckpoint puede pedir gradiente", (context) => {
    const dir = directory();
    context.after(() => rmSync(dir, { recursive: true, force: true }));
    const path = join(dir, "model.json");
    saveCheckpoint(path, { weights: Tensor.from([1], [1]) });
    const loaded = loadCheckpoint(path, { requiresGrad: true });
    assert.equal(loaded.parameters.weights.requiresGrad, true);
});

test("rechaza checkpoints inválidos", (context) => {
    const dir = directory();
    context.after(() => rmSync(dir, { recursive: true, force: true }));
    const base = { format: CHECKPOINT_FORMAT, version: CHECKPOINT_VERSION, metadata: {}, parameters: [] };
    const write = (name: string, value: object) => {
        const path = join(dir, name);
        writeFileSync(path, JSON.stringify(value), "utf8");
        return path;
    };

    assert.throws(() => loadCheckpoint(join(dir, "no-existe.json")), Error);
    assert.throws(
        () => loadCheckpoint(write("formato.json", { ...base, format: "otro" })),
        Error,
    );
    assert.throws(
        () => loadCheckpoint(write("version.json", { ...base, version: 2 })),
        Error,
    );
    assert.throws(
        () =>
            loadCheckpoint(
                write("forma.json", {
                    ...base,
                    parameters: [{ name: "w", shape: [2, 2], data: [1, 2, 3] }],
                }),
            ),
        Error,
    );
    assert.throws(
        () =>
            loadCheckpoint(
                write("datos.json", {
                    ...base,
                    parameters: [{ name: "w", shape: [1], data: [null] }],
                }),
            ),
        Error,
    );
    assert.throws(
        () =>
            loadCheckpoint(
                write("duplicado.json", {
                    ...base,
                    parameters: [
                        { name: "w", shape: [1], data: [1] },
                        { name: "w", shape: [1], data: [2] },
                    ],
                }),
            ),
        Error,
    );
});

test("saveCheckpoint rechaza valores no finitos", (context) => {
    const dir = directory();
    context.after(() => rmSync(dir, { recursive: true, force: true }));
    const path = join(dir, "model.json");
    const tensor = Tensor.from([1], [1]);
    tensor.data[0] = Number.NaN;
    assert.throws(() => saveCheckpoint(path, { w: tensor }), Error);
});
