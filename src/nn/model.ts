// Teoría: docs/teoria/entrenamiento/sets.md

import type { Tensor } from "../core/tensor.ts";

/** Contrato de un modelo entrenable: forward y parámetros para el optimizador. */
export interface Model {
    forward(input: Tensor): Tensor;
    parameters(): readonly Tensor[];
}
