// Teoría: docs/teoria/entrenamiento/sets.md

import type { Tensor } from "../core/tensor.ts";
import type { Model } from "../nn/model.ts";
import type { Optimizer } from "../optim/sgd.ts";

export interface Split {
    input: Tensor;
    target: Tensor;
}

export interface RegressionSet {
    name: string;
    description: string;
    kind: "regression";
    metricName: string;
    train: Split;
    validation: Split;
    test: Split;
    loss(prediction: Tensor, split: Split): Tensor;
    metric(prediction: Tensor, split: Split): number;
    baseline: number;
}

export interface Candidate {
    name: string;
    model: Model;
    optimizer: Optimizer;
    steps: number;
}

export interface Result {
    name: string;
    metric: string;
    value: number;
    baseline: number;
}
