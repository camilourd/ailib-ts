// Teoría: docs/teoria/entrenamiento/bucle.md
// Experimento: XOR con una capa oculta (2 → 4 ReLU → 2) y cross-entropy.

import { Tensor } from "../../src/core/tensor.ts";
import { relu } from "../../src/core/tensor-ops.ts";
import { crossEntropy } from "../../src/nn/losses.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { train } from "../../src/train/trainer.ts";

const inputs = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
const targets = [0, 1, 1, 0];
const hidden = new Linear(2, 4, { seed: 1 });
const output = new Linear(4, 2, { seed: 2 });
const optimizer = new SGD([...hidden.parameters(), ...output.parameters()], 0.5);

const history = train(
    (input) => output.forward(relu(hidden.forward(input))),
    inputs,
    {
        steps: 3000,
        optimizer,
        loss: (prediction) => crossEntropy(prediction, targets),
        onStep: (step, loss) => {
            if (step % 500 === 0) {
                console.log(`paso ${String(step).padStart(4)}  pérdida ${loss.toFixed(4)}`);
            }
        },
    },
);

const logits = output.forward(relu(hidden.forward(inputs)));
let correct = 0;
for (let row = 0; row < 4; row++) {
    const predicted = logits.get(row, 1) > logits.get(row, 0) ? 1 : 0;
    if (predicted === targets[row]) correct += 1;
}
console.log(`pérdida final ${(history.at(-1) as number).toFixed(4)} · aciertos ${correct}/4`);
