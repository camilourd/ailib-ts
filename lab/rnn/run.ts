// Teoría: docs/teoria/redes/rnn.md
// Experimento: una RNN procesa una secuencia y el gradiente llega a sus pesos.

import { Tensor } from "../../src/core/tensor.ts";
import { sum } from "../../src/core/tensor-ops.ts";
import { RNN } from "../../src/nn/rnn.ts";

const rnn = new RNN(3, 4, { seed: 1 });
const inputs = Array.from({ length: 5 }, (_, index) =>
    Tensor.from([index / 5, (index + 1) / 5, 1], [1, 3]),
);
const states = rnn.forward(inputs);
const last = states.at(-1) as Tensor;
sum(last).backward();

const norm = (data: Float32Array): number =>
    Math.sqrt([...data].reduce((total, value) => total + value * value, 0));

console.log(`estados: ${states.length} × ${JSON.stringify(states[0]?.shape)}`);
console.log(`norma del último estado: ${norm(last.data).toFixed(4)}`);
console.log(
    `norma del gradiente de los pesos: ` +
        rnn.parameters()
            .map((parameter) => norm(parameter.grad?.data ?? new Float32Array(0)).toFixed(4))
            .join(" · "),
);
