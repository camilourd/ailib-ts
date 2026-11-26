// Teoría: docs/teoria/redes/sigmoid.md
// Experimento: compara σ' analítica con la numérica y reporta el error máximo.

import { sigmoid, sigmoidDerivative } from "../../src/nn/sigmoid.ts";

const POINTS = [-6, -4, -2, -1, 0, 1, 2, 4, 6];
const H = 1e-6;

let worst = 0;
console.log("   x        σ(x)     σ'(x)   numérica     error");
for (const x of POINTS) {
    const analytic = sigmoidDerivative(x);
    const numeric = (sigmoid(x + H) - sigmoid(x - H)) / (2 * H);
    const error = Math.abs(numeric - analytic);
    worst = Math.max(worst, error);
    console.log(
        `${String(x).padStart(4)}  ${sigmoid(x).toFixed(6)}  ${analytic.toFixed(6)}  ` +
            `${numeric.toFixed(6)}  ${error.toExponential(2)}`,
    );
}
console.log(`error máximo: ${worst.toExponential(2)}`);
