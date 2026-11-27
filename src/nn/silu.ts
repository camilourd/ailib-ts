// Teoría: docs/teoria/secuencias/bloque-transformer.md

import { Tensor } from "../core/tensor.ts";
import { mul } from "../core/tensor-ops.ts";
import { sigmoidTensor } from "./sigmoid.ts";

/** SiLU(x) = x · σ(x), con gradiente. */
export function silu(input: Tensor): Tensor {
    return mul(input, sigmoidTensor(input));
}
