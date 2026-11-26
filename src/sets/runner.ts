// Teoría: docs/teoria/entrenamiento/sets.md

import { train } from "../train/trainer.ts";
import type { Candidate, RegressionSet, Result } from "./types.ts";

/** Entrena cada candidato en train y lo evalúa en test contra el baseline del set. */
export function benchmark(set: RegressionSet, candidates: readonly Candidate[]): Result[] {
    return candidates.map((candidate) => {
        train(
            (input) => candidate.model.forward(input),
            set.train.input,
            {
                steps: candidate.steps,
                optimizer: candidate.optimizer,
                loss: (prediction) => set.loss(prediction, set.train),
            },
        );
        const prediction = candidate.model.forward(set.test.input);
        return {
            name: candidate.name,
            metric: set.metricName,
            value: set.metric(prediction, set.test),
            baseline: set.baseline,
        };
    });
}
