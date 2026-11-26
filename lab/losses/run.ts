// Teoría: docs/teoria/entrenamiento/perdidas.md
// Experimento: clasificador lineal de 3 clases entrenado con cross-entropy y SGD.

import { Tensor } from "../../src/core/tensor.ts";
import { crossEntropy } from "../../src/nn/losses.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";

const input = Tensor.from([0, 0, 0, 1, 4, 0, 4, 1, 2, 4, 2, 5], [6, 2]);
const targets = [0, 0, 1, 1, 2, 2];
const layer = new Linear(2, 3, { seed: 1 });
const optimizer = new SGD(layer.parameters(), 0.5);

console.log("paso  pérdida");
for (let step = 0; step <= 200; step++) {
    optimizer.zeroGrad();
    const loss = crossEntropy(layer.forward(input), targets);
    loss.backward();
    optimizer.step();
    if (step % 50 === 0) {
        console.log(`${String(step).padStart(4)}  ${loss.get().toFixed(4)}`);
    }
}

const logits = layer.forward(input);
let correct = 0;
for (let row = 0; row < 6; row++) {
    let best = 0;
    for (let column = 1; column < 3; column++) {
        if (logits.get(row, column) > logits.get(row, best)) best = column;
    }
    if (best === targets[row]) correct += 1;
}
console.log(`aciertos: ${correct}/6`);
