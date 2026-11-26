// Teoría: docs/teoria/entrenamiento/sgd.md
// Experimento: minimiza f(x) = x² desde x = 4 con SGD y muestra la convergencia.

import { Tensor } from "../../src/core/tensor.ts";
import { mul } from "../../src/core/tensor-ops.ts";
import { SGD } from "../../src/optim/sgd.ts";

const x = Tensor.from([4], [1], { requiresGrad: true });
const optimizer = new SGD([x], 0.1);

console.log("paso        x        f(x) = x²");
for (let step = 0; step <= 20; step++) {
    if (step % 5 === 0) {
        const value = x.get(0);
        console.log(`${String(step).padStart(4)}  ${value.toFixed(6)}  ${(value * value).toFixed(6)}`);
    }
    optimizer.zeroGrad();
    mul(x, x).backward();
    optimizer.step();
}
