// Teoría: docs/teoria/nucleo/disjoint-set-union.md

/** Union-Find con compresión de caminos y unión por tamaño. */
export class DisjointSetUnion {
    #parents: Int32Array;
    #sizes: Int32Array;

    constructor(size = 0) {
        this.#parents = new Int32Array(0);
        this.#sizes = new Int32Array(0);
        this.restart(size);
    }

    /** Reinicia la estructura con `size` elementos aislados. */
    restart(size: number): void {
        if (!Number.isInteger(size) || size < 0) {
            throw new Error(`tamaño inválido: ${size}`);
        }
        this.#parents = new Int32Array(size);
        this.#sizes = new Int32Array(size);
        for (let index = 0; index < size; index++) {
            this.#parents[index] = index;
            this.#sizes[index] = 1;
        }
    }

    get size(): number {
        return this.#parents.length;
    }

    find(value: number): number {
        this.#check(value);
        let root = value;
        while ((this.#parents[root] as number) !== root) {
            root = this.#parents[root] as number;
        }
        let current = value;
        while ((this.#parents[current] as number) !== current) {
            const next = this.#parents[current] as number;
            this.#parents[current] = root;
            current = next;
        }
        return root;
    }

    join(first: number, second: number): void {
        let rootFirst = this.find(first);
        let rootSecond = this.find(second);
        if (rootFirst === rootSecond) return;
        if ((this.#sizes[rootFirst] as number) < (this.#sizes[rootSecond] as number)) {
            const swap = rootFirst;
            rootFirst = rootSecond;
            rootSecond = swap;
        }
        this.#parents[rootSecond] = rootFirst;
        this.#sizes[rootFirst] =
            (this.#sizes[rootFirst] as number) + (this.#sizes[rootSecond] as number);
    }

    #check(value: number): void {
        if (!Number.isInteger(value) || value < 0 || value >= this.size) {
            throw new RangeError(`índice ${value} fuera de [0, ${this.size})`);
        }
    }
}
