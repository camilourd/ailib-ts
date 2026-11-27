// Teoría: docs/teoria/entrenamiento/cuantizacion.md

import { Tensor } from "../core/tensor.ts";

export interface QuantizationOptions {
    perChannel?: boolean;
}

export interface QuantizedTensor {
    data: Int8Array;
    scales: Float32Array;
    shape: number[];
    perChannel: boolean;
}

/** Cuantización simétrica a INT8: q = round(x / escala), con escala = max|x| / 127. */
export function quantizeInt8(
    tensor: Tensor,
    options: QuantizationOptions = {},
): QuantizedTensor {
    if (tensor.rank !== 1 && tensor.rank !== 2) {
        throw new Error("la cuantización espera un vector o una matriz");
    }
    const perChannel = (options.perChannel ?? true) && tensor.rank === 2;
    const columns = tensor.rank === 2 ? (tensor.shape[1] as number) : 1;
    const rows = tensor.rank === 2 ? (tensor.shape[0] as number) : tensor.size;
    const scales = new Float32Array(perChannel ? columns : 1);
    const data = new Int8Array(tensor.size);

    for (let column = 0; column < (perChannel ? columns : 1); column++) {
        let maximum = 0;
        for (let row = 0; row < rows; row++) {
            const index = tensor.rank === 2 ? row * columns + column : row;
            maximum = Math.max(maximum, Math.abs(tensor.data[index] as number));
        }
        const scale = maximum > 0 ? maximum / 127 : 1;
        scales[column] = scale;
        for (let row = 0; row < rows; row++) {
            const index = tensor.rank === 2 ? row * columns + column : row;
            const quantized = Math.round((tensor.data[index] as number) / scale);
            data[index] = Math.max(-127, Math.min(127, quantized));
        }
    }

    return { data, scales, shape: [...tensor.shape], perChannel };
}

/** Reconstruye el tensor de punto flotante desde la representación INT8. */
export function dequantizeInt8(quantized: QuantizedTensor): Tensor {
    const columns = quantized.shape.length === 2 ? (quantized.shape[1] as number) : 1;
    const rows = quantized.shape.length === 2 ? (quantized.shape[0] as number) : quantized.shape[0];
    const data = new Float32Array(quantized.data.length);
    for (let row = 0; row < (rows as number); row++) {
        for (let column = 0; column < columns; column++) {
            const index = row * columns + column;
            const scale = quantized.perChannel
                ? (quantized.scales[column] as number)
                : (quantized.scales[0] as number);
            data[index] = (quantized.data[index] as number) * scale;
        }
    }
    return new Tensor(data, quantized.shape);
}

export function quantizedBytes(quantized: QuantizedTensor): number {
    return quantized.data.byteLength + quantized.scales.byteLength;
}

export function floatBytes(tensor: Tensor): number {
    return tensor.size * Float32Array.BYTES_PER_ELEMENT;
}
