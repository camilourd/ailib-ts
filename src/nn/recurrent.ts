// Teoría: docs/teoria/secuencias/gru-lstm.md

import type { Tensor } from "../core/tensor.ts";

export type CellType = "rnn" | "gru" | "lstm";

export interface RecurrentState {
    hidden: Tensor;
    cell?: Tensor;
}

export interface RecurrentCell {
    readonly inputSize: number;
    readonly hiddenSize: number;
    initial(): RecurrentState;
    advance(input: Tensor, state: RecurrentState): RecurrentState;
    parameters(): readonly Tensor[];
}
