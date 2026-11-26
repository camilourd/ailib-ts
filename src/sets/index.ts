// Teoría: docs/teoria/entrenamiento/sets.md

export { confusionMatrix, directionReport, errorsByDirection } from "./direction.ts";
export type { ConfusionMatrix, DirectionalErrors, DirectionReport } from "./direction.ts";
export {
    coefficientOfDetermination,
    correlation,
    directionalAccuracy,
    meanAbsoluteError,
    meanError,
    rootMeanSquaredError,
} from "./metrics.ts";
export { benchmark } from "./runner.ts";
export { syntheticAssetSet } from "./tabular/synthetic-asset.ts";
export type { Candidate, RegressionSet, Result, Split } from "./types.ts";
export type { Model } from "../nn/model.ts";
