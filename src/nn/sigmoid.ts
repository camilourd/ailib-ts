// Teoría: docs/teoria/redes/sigmoid.md

import { Tensor } from "../core/tensor.ts";
import { elementwiseRaw, unaryRaw } from "../core/tensor-ops.ts";

/** σ(x) = 1 / (1 + e^{-x}), evaluada sin desbordar el exponente. */
export function sigmoid(x: number): number {
    const e = Math.exp(-Math.abs(x));
    return x >= 0 ? 1 / (1 + e) : e / (1 + e);
}

/** σ'(x) = σ(x) · (1 − σ(x)). */
export function sigmoidDerivative(x: number): number {
    const value = sigmoid(x);
    return value * (1 - value);
}

/** σ elemento a elemento sobre un tensor; registra gradiente si hace falta. */
export function sigmoidTensor(tensor: Tensor): Tensor {
    const result = unaryRaw(tensor, sigmoid);
    if (tensor.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([tensor], (gradient) => {
            const derivative = unaryRaw(tensor, sigmoidDerivative);
            tensor.addGrad(elementwiseRaw(gradient, derivative, (left, right) => left * right));
        });
    }
    return result;
}
