// Teoría: docs/teoria/entrenamiento/checkpoints.md
// Experimento: entrena una capa, guarda el checkpoint, lo carga y comprueba que da las mismas salidas.

import { tmpdir } from "node:os";
import { join } from "node:path";

import { Tensor } from "../../src/core/tensor.ts";
import { loadCheckpoint, saveCheckpoint } from "../../src/io/checkpoint.ts";
import { mse } from "../../src/nn/losses.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";

const inputs = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
const targets = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
const layer = new Linear(2, 2, { seed: 1 });
const optimizer = new SGD(layer.parameters(), 0.5);

let lastLoss = 0;
for (let step = 0; step < 500; step++) {
    optimizer.zeroGrad();
    const loss = mse(layer.forward(inputs), targets);
    loss.backward();
    optimizer.step();
    lastLoss = loss.get();
}

const path = join(tmpdir(), "ailib-ts-checkpoint.json");
saveCheckpoint(
    path,
    { "layer.weights": layer.weights, "layer.bias": layer.bias },
    { model: "linear-identidad", sizes: [2, 2], step: 500, loss: lastLoss },
);

const loaded = loadCheckpoint(path);
const restored = new Linear(2, 2);
restored.weights.data.set(loaded.parameters["layer.weights"].data);
restored.bias.data.set(loaded.parameters["layer.bias"].data);

const before = layer.forward(inputs);
const after = restored.forward(inputs);
const same = [...before.data].every((value, index) => value === after.data[index]);

console.log(`checkpoint: ${path}`);
console.log(`metadata: ${JSON.stringify(loaded.metadata)}`);
console.log(`¿mismas salidas tras cargar? ${same ? "sí" : "no"}`);
