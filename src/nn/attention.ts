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
    const valueDimension = values.shape[1] as number;
    if (sequence === 0 || dimension === 0 || valueDimension === 0) {
        throw new Error("la atención necesita dimensiones positivas");
    }
    if ((keys.shape[0] as number) !== sequence || (keys.shape[1] as number) !== dimension) {
        throw new Error(
            `K debe ser [${sequence}, ${dimension}], no ${JSON.stringify(keys.shape)}`,
        );
    }
    if ((values.shape[0] as number) !== sequence) {
        throw new Error(`V debe tener ${sequence} filas, no ${values.shape[0]}`);
    }

    const causal = options.causal ?? true;
    const scale = 1 / Math.sqrt(dimension);
    const scores = new Float32Array(sequence * sequence);
    for (let row = 0; row < sequence; row++) {
        for (let column = 0; column < sequence; column++) {
            if (causal && column > row) {
                scores[row * sequence + column] = -Infinity;
            } else {
                let dot = 0;
                for (let index = 0; index < dimension; index++) {
                    dot += (queries.data[row * dimension + index] as number) *
                        (keys.data[column * dimension + index] as number);
                }
                scores[row * sequence + column] = dot * scale;
            }
        }
    }

    const weights = new Float32Array(sequence * sequence);
    for (let row = 0; row < sequence; row++) {
        let maximum = -Infinity;
        for (let column = 0; column < sequence; column++) {
            maximum = Math.max(maximum, scores[row * sequence + column] as number);
        }
        let total = 0;
        for (let column = 0; column < sequence; column++) {
            const value = Math.exp((scores[row * sequence + column] as number) - maximum);
            weights[row * sequence + column] = value;
            total += value;
        }
        for (let column = 0; column < sequence; column++) {
            weights[row * sequence + column] =
                (weights[row * sequence + column] as number) / total;
        }
    }

    const outputData = new Float32Array(sequence * valueDimension);
    for (let row = 0; row < sequence; row++) {
        for (let index = 0; index < valueDimension; index++) {
            let total = 0;
            for (let column = 0; column < sequence; column++) {
                total += (weights[row * sequence + column] as number) *
                    (values.data[column * valueDimension + index] as number);
            }
            outputData[row * valueDimension + index] = total;
        }
    }
    const output = new Tensor(outputData, [sequence, valueDimension]);
    const weightTensor = new Tensor(weights, [sequence, sequence]);
    if (queries.requiresGrad || keys.requiresGrad || values.requiresGrad) {
        output.requiresGrad = true;
        output.setGraph([queries, keys, values], (gradient) => {
            const dV = new Float32Array(sequence * valueDimension);
            for (let row = 0; row < sequence; row++) {
                for (let column = 0; column < sequence; column++) {
                    const weight = weights[row * sequence + column] as number;
                    for (let index = 0; index < valueDimension; index++) {
                        dV[column * valueDimension + index] =
                            (dV[column * valueDimension + index] as number) +
                            weight * (gradient.data[row * valueDimension + index] as number);
                    }
                }
            }

            const dP = new Float32Array(sequence * sequence);
            for (let row = 0; row < sequence; row++) {
                for (let column = 0; column < sequence; column++) {
                    let total = 0;
                    for (let index = 0; index < valueDimension; index++) {
                        total += (gradient.data[row * valueDimension + index] as number) *
                            (values.data[column * valueDimension + index] as number);
                    }
                    dP[row * sequence + column] = total;
                }
            }

            const dS = new Float32Array(sequence * sequence);
            for (let row = 0; row < sequence; row++) {
                let rowDot = 0;
                for (let column = 0; column < sequence; column++) {
                    rowDot += (dP[row * sequence + column] as number) *
                        (weights[row * sequence + column] as number);
                }
                for (let column = 0; column < sequence; column++) {
                    if (!causal || column <= row) {
                        dS[row * sequence + column] =
                            (weights[row * sequence + column] as number) *
                            ((dP[row * sequence + column] as number) - rowDot);
                    }
                }
            }

            const dQ = new Float32Array(sequence * dimension);
            const dK = new Float32Array(sequence * dimension);
            for (let row = 0; row < sequence; row++) {
                for (let index = 0; index < dimension; index++) {
                    let total = 0;
                    for (let column = 0; column < sequence; column++) {
                        total += (dS[row * sequence + column] as number) *
                            (keys.data[column * dimension + index] as number);
                    }
                    dQ[row * dimension + index] = total * scale;
                }
            }
            for (let column = 0; column < sequence; column++) {
                for (let index = 0; index < dimension; index++) {
                    let total = 0;
                    for (let row = 0; row < sequence; row++) {
                        total += (dS[row * sequence + column] as number) *
                            (queries.data[row * dimension + index] as number);
                    }
                    dK[column * dimension + index] = total * scale;
                }
            }

            queries.addGrad(new Tensor(dQ, [sequence, dimension]));
            keys.addGrad(new Tensor(dK, [sequence, dimension]));
            values.addGrad(new Tensor(dV, [sequence, valueDimension]));
        });
    }
    return { output, weights: weightTensor };
}
