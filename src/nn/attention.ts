// Teoría: docs/teoria/secuencias/atencion.md

import { Tensor } from "../core/tensor.ts";

export interface AttentionOptions {
    causal?: boolean;
}

export interface AttentionResult {
    output: Tensor;
    weights: Tensor;
}

/**
 * Atención escalada de un solo ejemplo: scores = QK^T / sqrt(d), softmax por filas y salida P·V.
 * Con `causal: true`, cada posición solo puede mirar hacia atrás.
 */
export function scaledDotProductAttention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    options: AttentionOptions = {},
): AttentionResult {
    if (queries.rank !== 2 || keys.rank !== 2 || values.rank !== 2) {
        throw new Error("Q, K y V deben ser matrices [secuencia, dimensión]");
    }
    const sequence = queries.shape[0] as number;
    const dimension = queries.shape[1] as number;
    const keySequence = keys.shape[0] as number;
    const valueDimension = values.shape[1] as number;
    if (sequence === 0 || dimension === 0 || keySequence === 0 || valueDimension === 0) {
        throw new Error("la atención necesita dimensiones positivas");
    }
    if ((keys.shape[1] as number) !== dimension) {
        throw new Error(
            `K debe ser [${keySequence}, ${dimension}], no ${JSON.stringify(keys.shape)}`,
        );
    }
    if ((values.shape[0] as number) !== keySequence) {
        throw new Error(`V debe tener ${keySequence} filas, no ${values.shape[0]}`);
    }

    const causal = options.causal ?? true;
    const scale = 1 / Math.sqrt(dimension);
    const scores = new Float32Array(sequence * keySequence);
    for (let row = 0; row < sequence; row++) {
        for (let column = 0; column < keySequence; column++) {
            if (causal && column > row) {
                scores[row * keySequence + column] = -Infinity;
            } else {
                let dot = 0;
                for (let index = 0; index < dimension; index++) {
                    dot += (queries.data[row * dimension + index] as number) *
                        (keys.data[column * dimension + index] as number);
                }
                scores[row * keySequence + column] = dot * scale;
            }
        }
    }

    const weights = new Float32Array(sequence * keySequence);
    for (let row = 0; row < sequence; row++) {
        let maximum = -Infinity;
        for (let column = 0; column < keySequence; column++) {
            maximum = Math.max(maximum, scores[row * keySequence + column] as number);
        }
        let total = 0;
        for (let column = 0; column < keySequence; column++) {
            const value = Math.exp((scores[row * keySequence + column] as number) - maximum);
            weights[row * keySequence + column] = value;
            total += value;
        }
        for (let column = 0; column < keySequence; column++) {
            weights[row * keySequence + column] =
                (weights[row * keySequence + column] as number) / total;
        }
    }

    const outputData = new Float32Array(sequence * valueDimension);
    for (let row = 0; row < sequence; row++) {
        for (let index = 0; index < valueDimension; index++) {
            let total = 0;
            for (let column = 0; column < keySequence; column++) {
                total += (weights[row * keySequence + column] as number) *
                    (values.data[column * valueDimension + index] as number);
            }
            outputData[row * valueDimension + index] = total;
        }
    }
    const output = new Tensor(outputData, [sequence, valueDimension]);
    const weightTensor = new Tensor(weights, [sequence, keySequence]);
    if (queries.requiresGrad || keys.requiresGrad || values.requiresGrad) {
        output.requiresGrad = true;
        output.setGraph([queries, keys, values], (gradient) => {
            const dV = new Float32Array(keySequence * valueDimension);
            for (let row = 0; row < sequence; row++) {
                for (let column = 0; column < keySequence; column++) {
                    const weight = weights[row * keySequence + column] as number;
                    for (let index = 0; index < valueDimension; index++) {
                        dV[column * valueDimension + index] =
                            (dV[column * valueDimension + index] as number) +
                            weight * (gradient.data[row * valueDimension + index] as number);
                    }
                }
            }

            const dP = new Float32Array(sequence * keySequence);
            for (let row = 0; row < sequence; row++) {
                for (let column = 0; column < keySequence; column++) {
                    let total = 0;
                    for (let index = 0; index < valueDimension; index++) {
                        total += (gradient.data[row * valueDimension + index] as number) *
                            (values.data[column * valueDimension + index] as number);
                    }
                    dP[row * keySequence + column] = total;
                }
            }

            const dS = new Float32Array(sequence * keySequence);
            for (let row = 0; row < sequence; row++) {
                let rowDot = 0;
                for (let column = 0; column < keySequence; column++) {
                    rowDot += (dP[row * keySequence + column] as number) *
                        (weights[row * keySequence + column] as number);
                }
                for (let column = 0; column < keySequence; column++) {
                    if (!causal || column <= row) {
                        dS[row * keySequence + column] =
                            (weights[row * keySequence + column] as number) *
                            ((dP[row * keySequence + column] as number) - rowDot);
                    }
                }
            }

            const dQ = new Float32Array(sequence * dimension);
            const dK = new Float32Array(keySequence * dimension);
            for (let row = 0; row < sequence; row++) {
                for (let index = 0; index < dimension; index++) {
                    let total = 0;
                    for (let column = 0; column < keySequence; column++) {
                        total += (dS[row * keySequence + column] as number) *
                            (keys.data[column * dimension + index] as number);
                    }
                    dQ[row * dimension + index] = total * scale;
                }
            }
            for (let column = 0; column < keySequence; column++) {
                for (let index = 0; index < dimension; index++) {
                    let total = 0;
                    for (let row = 0; row < sequence; row++) {
                        total += (dS[row * keySequence + column] as number) *
                            (queries.data[row * dimension + index] as number);
                    }
                    dK[column * dimension + index] = total * scale;
                }
            }

            queries.addGrad(new Tensor(dQ, [sequence, dimension]));
            keys.addGrad(new Tensor(dK, [keySequence, dimension]));
            values.addGrad(new Tensor(dV, [keySequence, valueDimension]));
        });
    }
    return { output, weights: weightTensor };
}
