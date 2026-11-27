// Teoría: docs/experimentos/evolucion-tokenizers.md

interface TrieNode {
    children: Map<number, TrieNode>;
    token?: number;
}

export interface ContextTokenizerOptions {
    leftContext?: number;
    rightContext?: number;
    lookahead?: "chars" | "word";
    penalty?: number;
}

type ByteClass = "letter" | "digit" | "space" | "other";

function classOf(byte: number): ByteClass {
    if (byte === 32 || byte === 9 || byte === 10 || byte === 13) return "space";
    if (byte >= 48 && byte <= 57) return "digit";
    if ((byte >= 65 && byte <= 90) || (byte >= 97 && byte <= 122) || byte >= 128) return "letter";
    return "other";
}

/**
 * Tokenizer lossless con segmentación por programación dinámica. El coste de un corte depende del
 * contexto izquierdo/derecho: partir una palabra se penaliza; `lookahead: "word"` mira hasta el final
 * de la palabra siguiente.
 */
export class ContextTokenizer {
    readonly #tokens: Uint8Array[];
    readonly #root: TrieNode = { children: new Map() };
    readonly #left: number;
    readonly #right: number;
    readonly #lookahead: "chars" | "word";
    readonly #penalty: number;

    constructor(tokens: readonly Uint8Array[], options: ContextTokenizerOptions = {}) {
        if (tokens.length === 0) throw new Error("hacen falta tokens");
        this.#tokens = [...tokens];
        this.#left = options.leftContext ?? 2;
        this.#right = options.rightContext ?? 2;
        this.#lookahead = options.lookahead ?? "chars";
        this.#penalty = options.penalty ?? 0.5;
        if (!Number.isInteger(this.#left) || this.#left < 0 || this.#left > 3) {
            throw new Error(`leftContext inválido: ${this.#left}`);
        }
        if (!Number.isInteger(this.#right) || this.#right < 0 || this.#right > 3) {
            throw new Error(`rightContext inválido: ${this.#right}`);
        }
        if (!Number.isFinite(this.#penalty) || this.#penalty < 0) {
            throw new Error(`penalty inválida: ${this.#penalty}`);
        }
        for (let id = 0; id < this.#tokens.length; id++) {
            let node = this.#root;
            for (const byte of this.#tokens[id] as Uint8Array) {
                let child = node.children.get(byte);
                if (!child) {
                    child = { children: new Map() };
                    node.children.set(byte, child);
                }
                node = child;
            }
            node.token = id;
        }
    }

    get vocabularySize(): number {
        return this.#tokens.length;
    }

    encode(text: string): number[] {
        const bytes = new TextEncoder().encode(text);
        const size = bytes.length;
        const costs = new Float64Array(size + 1).fill(Number.POSITIVE_INFINITY);
        const previous = new Int32Array(size + 1).fill(-1);
        const chosen = new Int32Array(size + 1).fill(-1);
        costs[0] = 0;
        for (let position = 0; position < size; position++) {
            const base = costs[position] as number;
            if (Number.isFinite(base)) {
                let node = this.#root;
                let end = position;
                let searching = true;
                while (end < size && searching) {
                    const child = node.children.get(bytes[end] as number);
                    if (!child) {
                        searching = false;
                    } else {
                        node = child;
                        if (node.token !== undefined) {
                            const cost = base + 1 + this.#boundaryPenalty(bytes, position, end + 1);
                            if (cost < (costs[end + 1] as number)) {
                                costs[end + 1] = cost;
                                previous[end + 1] = position;
                                chosen[end + 1] = node.token;
                            }
                        }
                        end += 1;
                    }
                }
            }
        }
        if (!Number.isFinite(costs[size] as number)) {
            throw new Error("no hay segmentación con los tokens dados");
        }
        const tokens: number[] = [];
        let cursor = size;
        while (cursor > 0) {
            tokens.push(chosen[cursor] as number);
            cursor = previous[cursor] as number;
        }
        return tokens.reverse();
    }

    decode(tokens: readonly number[]): string {
        const bytes: number[] = [];
        for (const token of tokens) {
            if (!Number.isInteger(token) || token < 0 || token >= this.#tokens.length) {
                throw new RangeError(`token fuera de vocabulario: ${token}`);
            }
            for (const byte of this.#tokens[token] as Uint8Array) bytes.push(byte);
        }
        return new TextDecoder().decode(Uint8Array.from(bytes));
    }

    #boundaryPenalty(bytes: Uint8Array, start: number, end: number): number {
        let penalty = 0;
        const first = classOf(bytes[start] as number);
        const last = classOf(bytes[end - 1] as number);
        if (this.#left > 0 && start > 0 && first === "letter") {
            const before = classOf(bytes[start - 1] as number);
            if (before === "letter") penalty += this.#penalty * (this.#left / 3);
        }
        if (this.#right > 0 && end < bytes.length && last === "letter") {
            const after = classOf(bytes[end] as number);
            if (after === "letter") penalty += this.#penalty * (this.#right / 3);
        }
        if (this.#lookahead === "word" && end < bytes.length && last === "letter") {
            let wordEnd = end;
            while (wordEnd < bytes.length && classOf(bytes[wordEnd] as number) === "letter") {
                wordEnd += 1;
            }
            if (wordEnd > end) penalty += this.#penalty * ((wordEnd - end) / Math.max(1, wordEnd - start));
        }
        return penalty;
    }
}
