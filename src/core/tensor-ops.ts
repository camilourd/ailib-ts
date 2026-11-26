// Teoría: docs/teoria/nucleo/broadcasting.md · autograd: docs/teoria/nucleo/autograd.md

import { Tensor } from "./tensor.ts";

/** Forma resultante de combinar dos formas con las reglas de broadcasting. */
export function broadcastShapes(left: readonly number[], right: readonly number[]): number[] {
    const rank = Math.max(left.length, right.length);
    const shape = new Array<number>(rank);
    for (let axis = 0; axis < rank; axis++) {
        const leftDimension = left[left.length - 1 - axis] ?? 1;
        const rightDimension = right[right.length - 1 - axis] ?? 1;
        if (leftDimension === rightDimension) {
            shape[rank - 1 - axis] = leftDimension;
        } else if (leftDimension === 1) {
            shape[rank - 1 - axis] = rightDimension;
        } else if (rightDimension === 1) {
            shape[rank - 1 - axis] = leftDimension;
        } else {
            throw new Error(
                `formas incompatibles para broadcasting: ${JSON.stringify(left)} y ${JSON.stringify(right)}`,
            );
        }
    }
    return shape;
}

/** Strides del operando `shape` vistos desde `target`; 0 en las dimensiones expandidas. */
export function broadcastStrides(
    shape: readonly number[],
    strides: readonly number[],
    target: readonly number[],
): number[] {
    const result = new Array<number>(target.length).fill(0);
    for (let axis = 0; axis < target.length; axis++) {
        const sourceAxis = shape.length - (target.length - axis);
        if (sourceAxis >= 0) {
            const dimension = shape[sourceAxis] as number;
            if (dimension === target[axis]) {
                result[axis] = strides[sourceAxis] as number;
            } else if (dimension !== 1) {
                throw new Error(`la dimensión ${sourceAxis} (${dimension}) no encaja en ${target[axis]}`);
            }
        }
    }
    return result;
}

/** Suma el gradiente sobre las dimensiones que el operando había expandido. */
function unbroadcast(gradient: Tensor, shape: readonly number[]): Tensor {
    const sameShape =
        gradient.shape.length === shape.length &&
        gradient.shape.every((dimension, axis) => dimension === shape[axis]);
    if (sameShape) return gradient;
    const result = Tensor.zeros(shape);
    for (let linear = 0; linear < gradient.size; linear++) {
        let remainder = linear;
        let offset = 0;
        for (let axis = 0; axis < gradient.rank; axis++) {
            const stride = gradient.strides[axis] as number;
            const index = Math.floor(remainder / stride);
            remainder -= index * stride;
            const targetAxis = shape.length - (gradient.rank - axis);
            if (targetAxis >= 0) {
                const targetIndex = (shape[targetAxis] as number) === 1 ? 0 : index;
                offset += targetIndex * (result.strides[targetAxis] as number);
            }
        }
        result.data[offset] = (result.data[offset] as number) + (gradient.data[linear] as number);
    }
    return result;
}

/** Expande un gradiente a `shape` (el inverso de reducir un eje). */
function broadcastTo(gradient: Tensor, shape: readonly number[]): Tensor {
    const result = Tensor.zeros(shape);
    const strides = broadcastStrides(gradient.shape, gradient.strides, shape);
    for (let linear = 0; linear < result.size; linear++) {
        let remainder = linear;
        let offset = 0;
        for (let axis = 0; axis < shape.length; axis++) {
            const stride = result.strides[axis] as number;
            const index = Math.floor(remainder / stride);
            remainder -= index * stride;
            offset += index * (strides[axis] as number);
        }
        result.data[linear] = gradient.data[offset] as number;
    }
    return result;
}

/** Multiplica por un escalar sin registrar grafo. */
function scaleRaw(tensor: Tensor, factor: number): Tensor {
    const result = Tensor.zeros(tensor.shape);
    for (let index = 0; index < tensor.size; index++) {
        result.data[index] = (tensor.data[index] as number) * factor;
    }
    return result;
}

/** Operación elemento a elemento sin registrar grafo. Interno de las operaciones. */
export function elementwiseRaw(
    left: Tensor,
    right: Tensor,
    combine: (leftValue: number, rightValue: number) => number,
): Tensor {
    const shape = broadcastShapes(left.shape, right.shape);
    const leftStrides = broadcastStrides(left.shape, left.strides, shape);
    const rightStrides = broadcastStrides(right.shape, right.strides, shape);
    const result = Tensor.zeros(shape);
    for (let linear = 0; linear < result.size; linear++) {
        let remainder = linear;
        let leftOffset = 0;
        let rightOffset = 0;
        for (let axis = 0; axis < shape.length; axis++) {
            const stride = result.strides[axis] as number;
            const index = Math.floor(remainder / stride);
            remainder -= index * stride;
            leftOffset += index * (leftStrides[axis] as number);
            rightOffset += index * (rightStrides[axis] as number);
        }
        result.data[linear] = combine(
            left.data[leftOffset] as number,
            right.data[rightOffset] as number,
        );
    }
    return result;
}

/** Aplica una función elemento a elemento sin registrar grafo. Interno de las operaciones. */
export function unaryRaw(tensor: Tensor, apply: (value: number) => number): Tensor {
    const result = Tensor.zeros(tensor.shape);
    for (let index = 0; index < tensor.size; index++) {
        result.data[index] = apply(tensor.data[index] as number);
    }
    return result;
}

function unaryWithGrad(
    tensor: Tensor,
    apply: (value: number) => number,
    derivative: (value: number) => number,
): Tensor {
    const result = unaryRaw(tensor, apply);
    if (tensor.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([tensor], (gradient) => {
            const local = unaryRaw(tensor, derivative);
            tensor.addGrad(elementwiseRaw(gradient, local, (left, right) => left * right));
        });
    }
    return result;
}

/** e^x elemento a elemento. */
export function exp(tensor: Tensor): Tensor {
    return unaryWithGrad(tensor, Math.exp, Math.exp);
}

/** Logaritmo natural elemento a elemento; `log(0)` es −Infinity. */
export function log(tensor: Tensor): Tensor {
    return unaryWithGrad(tensor, Math.log, (value) => 1 / value);
}

/** ReLU: max(0, x). */
export function relu(tensor: Tensor): Tensor {
    return unaryWithGrad(tensor, (value) => Math.max(0, value), (value) => (value > 0 ? 1 : 0));
}

/** Suma elemento a elemento, con broadcasting. */
export function add(left: Tensor, right: Tensor): Tensor {
    const result = elementwiseRaw(left, right, (leftValue, rightValue) => leftValue + rightValue);
    if (left.requiresGrad || right.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([left, right], (gradient) => {
            left.addGrad(unbroadcast(gradient, left.shape));
            right.addGrad(unbroadcast(gradient, right.shape));
        });
    }
    return result;
}

/** Resta elemento a elemento, con broadcasting. */
export function sub(left: Tensor, right: Tensor): Tensor {
    const result = elementwiseRaw(left, right, (leftValue, rightValue) => leftValue - rightValue);
    if (left.requiresGrad || right.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([left, right], (gradient) => {
            left.addGrad(unbroadcast(gradient, left.shape));
            right.addGrad(unbroadcast(scaleRaw(gradient, -1), right.shape));
        });
    }
    return result;
}

/** Producto elemento a elemento, con broadcasting. */
export function mul(left: Tensor, right: Tensor): Tensor {
    const result = elementwiseRaw(left, right, (leftValue, rightValue) => leftValue * rightValue);
    if (left.requiresGrad || right.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([left, right], (gradient) => {
            left.addGrad(
                unbroadcast(elementwiseRaw(gradient, right, (value, other) => value * other), left.shape),
            );
            right.addGrad(
                unbroadcast(elementwiseRaw(gradient, left, (value, other) => value * other), right.shape),
            );
        });
    }
    return result;
}

function matmulRaw(left: Tensor, right: Tensor): Tensor {
    if (left.rank !== 2 || right.rank !== 2) {
        throw new Error("matmul espera dos matrices de rango 2");
    }
    const rows = left.shape[0] as number;
    const inner = left.shape[1] as number;
    const columns = right.shape[1] as number;
    if (inner !== (right.shape[0] as number)) {
        throw new Error(
            `formas incompatibles para matmul: ${JSON.stringify(left.shape)} y ${JSON.stringify(right.shape)}`,
        );
    }
    const result = Tensor.zeros([rows, columns]);
    const leftRowStride = left.strides[0] as number;
    const leftColumnStride = left.strides[1] as number;
    const rightRowStride = right.strides[0] as number;
    const rightColumnStride = right.strides[1] as number;
    for (let row = 0; row < rows; row++) {
        for (let column = 0; column < columns; column++) {
            let total = 0;
            for (let step = 0; step < inner; step++) {
                total +=
                    (left.data[row * leftRowStride + step * leftColumnStride] as number) *
                    (right.data[step * rightRowStride + column * rightColumnStride] as number);
            }
            result.data[row * columns + column] = total;
        }
    }
    return result;
}

/** Producto matricial 2D: [m, k] · [k, n] → [m, n]. */
export function matmul(left: Tensor, right: Tensor): Tensor {
    const result = matmulRaw(left, right);
    if (left.requiresGrad || right.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([left, right], (gradient) => {
            left.addGrad(matmulRaw(gradient, right.transpose()));
            right.addGrad(matmulRaw(left.transpose(), gradient));
        });
    }
    return result;
}

/** Suma todos los elementos o los de un eje (ese eje desaparece del resultado). */
export function sum(tensor: Tensor, axis?: number): Tensor {
    const reducedAxis = axis;
    let result: Tensor;
    if (reducedAxis === undefined) {
        let total = 0;
        for (const value of tensor.data) total += value;
        result = Tensor.from([total], []);
    } else {
        if (!Number.isInteger(reducedAxis) || reducedAxis < 0 || reducedAxis >= tensor.rank) {
            throw new RangeError(`eje ${reducedAxis} fuera de [0, ${tensor.rank})`);
        }
        const shape = tensor.shape.filter((_, index) => index !== reducedAxis);
        result = Tensor.zeros(shape);
        for (let linear = 0; linear < tensor.size; linear++) {
            let remainder = linear;
            let outputOffset = 0;
            for (let sourceAxis = 0; sourceAxis < tensor.rank; sourceAxis++) {
                const stride = tensor.strides[sourceAxis] as number;
                const index = Math.floor(remainder / stride);
                remainder -= index * stride;
                if (sourceAxis !== reducedAxis) {
                    const outputAxis = sourceAxis < reducedAxis ? sourceAxis : sourceAxis - 1;
                    outputOffset += index * (result.strides[outputAxis] as number);
                }
            }
            result.data[outputOffset] = (result.data[outputOffset] as number) + (tensor.data[linear] as number);
        }
    }
    if (tensor.requiresGrad) {
        result.requiresGrad = true;
        result.setGraph([tensor], (gradient) => {
            const expandedShape =
                reducedAxis === undefined
                    ? tensor.shape
                    : tensor.shape.map((dimension, index) => (index === reducedAxis ? 1 : dimension));
            const expanded = reducedAxis === undefined ? gradient : gradient.reshape(expandedShape);
            tensor.addGrad(broadcastTo(expanded, tensor.shape));
        });
    }
    return result;
}

/** Media de todos los elementos o de un eje: `sum` dividido por cuántos sumó. */
export function mean(tensor: Tensor, axis?: number): Tensor {
    const total = sum(tensor, axis);
    const count = axis === undefined ? tensor.size : (tensor.shape[axis] as number);
    return mul(total, Tensor.from([1 / count], []));
}
