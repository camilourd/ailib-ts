// Teoría: docs/teoria/secuencias/rope.md
// Experimento: un vector 2D gira según su posición y el producto q·k depende de la distancia.

import { Tensor } from "../../src/core/tensor.ts";
import { rope } from "../../src/nn/rope.ts";

const vector = Tensor.from([1, 0], [1, 2]);
const dot = (first: Tensor, second: Tensor): number =>
    (first.data[0] as number) * (second.data[0] as number) +
    (first.data[1] as number) * (second.data[1] as number);

for (const position of [0, 1, 2, 3]) {
    const rotated = rope(vector, { offset: position });
    console.log(
        `posición ${position}: [${(rotated.data[0] as number).toFixed(3)}, ` +
            `${(rotated.data[1] as number).toFixed(3)}]`,
    );
}

const query = Tensor.from([1, 0.5], [1, 2]);
const key = Tensor.from([0.3, -0.7], [1, 2]);
console.log(
    `q·k a distancia 2: ` +
        `${dot(rope(query, { offset: 3 }), rope(key, { offset: 5 })).toFixed(5)}`,
);
console.log(
    `q·k a distancia 2 (otro origen): ` +
        `${dot(rope(query, { offset: 0 }), rope(key, { offset: 2 })).toFixed(5)}`,
);
