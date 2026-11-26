// Teoría: docs/teoria/nucleo/tensor.md

const isDimension = (value: number): boolean => Number.isInteger(value) && value >= 0;

const sizeOf = (shape: readonly number[]): number =>
    shape.reduce((total, dimension) => total * dimension, 1);

/** Strides canónicos (row-major): el último eje avanza de uno en uno. */
const stridesOf = (shape: readonly number[]): number[] => {
    const strides = new Array<number>(shape.length).fill(1);
    let stride = 1;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
        strides[axis] = stride;
        stride *= shape[axis] as number;
    }
    return strides;
};

const checkShape = (shape: readonly number[]): void => {
    if (!shape.every(isDimension)) {
        throw new Error(`forma inválida: ${JSON.stringify(shape)}; cada dimensión debe ser un entero ≥ 0`);
    }
};

export interface TensorOptions {
    requiresGrad?: boolean;
}

/**
 * Tensor n-dimensional sobre un buffer plano (f32), con forma y strides canónicos. Sin vistas:
 * `reshape` y `transpose` devuelven tensores nuevos con sus propios datos.
 */
export class Tensor {
    readonly data: Float32Array;
    readonly shape: readonly number[];
    readonly strides: readonly number[];
    readonly size: number;
    requiresGrad: boolean;
    grad: Tensor | null = null;
    #parents: readonly Tensor[] = [];
    #backward: ((gradient: Tensor) => void) | null = null;

    constructor(data: Float32Array, shape: readonly number[], options: TensorOptions = {}) {
        checkShape(shape);
        const size = sizeOf(shape);
        if (data.length !== size) {
            throw new Error(
                `los datos tienen ${data.length} elementos y la forma ${JSON.stringify(shape)} pide ${size}`,
            );
        }
        this.data = data;
        this.shape = [...shape];
        this.strides = stridesOf(shape);
        this.size = size;
        this.requiresGrad = options.requiresGrad ?? false;
    }

    get rank(): number {
        return this.shape.length;
    }

    static zeros(shape: readonly number[], options: TensorOptions = {}): Tensor {
        checkShape(shape);
        return new Tensor(new Float32Array(sizeOf(shape)), shape, options);
    }

    static ones(shape: readonly number[], options: TensorOptions = {}): Tensor {
        const tensor = Tensor.zeros(shape, options);
        tensor.data.fill(1);
        return tensor;
    }

    static from(values: ArrayLike<number>, shape?: readonly number[], options: TensorOptions = {}): Tensor {
        const data = Float32Array.from(values);
        return new Tensor(data, shape ?? [data.length], options);
    }

    get(...indices: number[]): number {
        return this.data[this.offsetOf(indices)] as number;
    }

    set(value: number, ...indices: number[]): void {
        this.data[this.offsetOf(indices)] = value;
    }

    reshape(shape: readonly number[]): Tensor {
        checkShape(shape);
        const size = sizeOf(shape);
        if (size !== this.size) {
            throw new Error(
                `no se puede pasar de ${JSON.stringify(this.shape)} (${this.size}) a ${JSON.stringify(shape)} (${size})`,
            );
        }
        return new Tensor(this.data.slice(), shape);
    }

    transpose(axes?: readonly number[]): Tensor {
        const order = axes ?? [...this.shape.keys()].reverse();
        if (order.length !== this.rank) {
            throw new Error(`el orden debe tener ${this.rank} ejes, no ${order.length}`);
        }
        const seen = new Set<number>();
        for (const axis of order) {
            if (!Number.isInteger(axis) || axis < 0 || axis >= this.rank || seen.has(axis)) {
                throw new Error(`orden inválido: ${JSON.stringify(order)}`);
            }
            seen.add(axis);
        }
        const shape = order.map((axis) => this.shape[axis] as number);
        const result = new Tensor(new Float32Array(this.size), shape);
        for (let linear = 0; linear < this.size; linear++) {
            let remainder = linear;
            let sourceOffset = 0;
            for (let axis = 0; axis < order.length; axis++) {
                const stride = result.strides[axis] as number;
                const index = Math.floor(remainder / stride);
                remainder -= index * stride;
                sourceOffset += index * (this.strides[order[axis] as number] as number);
            }
            result.data[linear] = this.data[sourceOffset] as number;
        }
        return result;
    }

    /** Engancha el resultado al grafo. Interno de las operaciones. */
    setGraph(parents: readonly Tensor[], backward: (gradient: Tensor) => void): void {
        this.#parents = parents;
        this.#backward = backward;
    }

    /** Acumula un gradiente con la forma de este tensor. Interno de las operaciones. */
    addGrad(gradient: Tensor): void {
        if (!this.requiresGrad) return;
        const sameShape =
            gradient.shape.length === this.shape.length &&
            gradient.shape.every((dimension, axis) => dimension === this.shape[axis]);
        if (!sameShape) {
            throw new Error("el gradiente no tiene la forma del tensor");
        }
        if (this.grad === null) {
            this.grad = new Tensor(gradient.data.slice(), gradient.shape);
            return;
        }
        for (let index = 0; index < this.grad.data.length; index++) {
            this.grad.data[index] = (this.grad.data[index] as number) + (gradient.data[index] as number);
        }
    }

    zeroGrad(): void {
        this.grad = null;
    }

    /** Retropropaga desde este tensor (semilla de unos) por todo el grafo. */
    backward(): void {
        const order: Tensor[] = [];
        const visited = new Set<Tensor>();
        const visit = (tensor: Tensor): void => {
            if (visited.has(tensor)) return;
            visited.add(tensor);
            for (const parent of tensor.#parents) visit(parent);
            order.push(tensor);
        };
        visit(this);
        this.grad = Tensor.ones(this.shape);
        for (let index = order.length - 1; index >= 0; index--) {
            const tensor = order[index] as Tensor;
            const gradient = tensor.grad;
            if (gradient && tensor.#backward) tensor.#backward(gradient);
        }
    }

    private offsetOf(indices: readonly number[]): number {
        if (indices.length !== this.rank) {
            throw new Error(`se esperaban ${this.rank} índices, no ${indices.length}`);
        }
        let offset = 0;
        for (let axis = 0; axis < indices.length; axis++) {
            const index = indices[axis] as number;
            const dimension = this.shape[axis] as number;
            if (!Number.isInteger(index) || index < 0 || index >= dimension) {
                throw new RangeError(`índice ${index} fuera de [0, ${dimension}) en el eje ${axis}`);
            }
            offset += index * (this.strides[axis] as number);
        }
        return offset;
    }
}
