// Teoría: docs/teoria/nucleo/autograd.md
// Experimento: deriva f(x) = (x + x) * x = 2x² y comprueba que el gradiente es 4x.

import { Tensor } from "../../src/core/tensor.ts";
import { add, mul } from "../../src/core/tensor-ops.ts";

const input = Tensor.from([1, 2, 3], [3], { requiresGrad: true });
const output = mul(add(input, input), input);
output.backward();

console.log("f(x) = (x + x) * x = 2x²");
console.log(`x = [${[...input.data].join(", ")}]`);
console.log(`gradiente = [${[...(input.grad?.data ?? [])].join(", ")}] (esperado 4x = [4, 8, 12])`);
