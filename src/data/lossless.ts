// Teoría: docs/experimentos/evolucion-tokenizers.md

interface TrieNode {
    children: Map<number, TrieNode>;
    token?: number;
}

/**
 * Tokenizer cuyos tokens son secuencias de bytes. Empieza con los 256 bytes, así que siempre puede
 * reconstruir cualquier texto; los tokens adicionales solo acortan la secuencia.
 */
export class LosslessByteTokenizer {
    readonly #tokens: Uint8Array[] = [];
    readonly #ids = new Map<string, number>();
    readonly #root: TrieNode = { children: new Map() };
    readonly #maxTokens: number;

    constructor(maxTokens = 4096) {
        if (!Number.isInteger(maxTokens) || maxTokens < 256) {
            throw new Error(`maxTokens debe ser un entero ≥ 256, no ${maxTokens}`);
        }
        this.#maxTokens = maxTokens;
        for (let byte = 0; byte < 256; byte++) {
            this.add(new Uint8Array([byte]));
        }
    }

    get vocabularySize(): number {
        return this.#tokens.length;
    }

    add(bytes: Uint8Array): number | null {
        if (bytes.length === 0) throw new Error("un token no puede estar vacío");
        const key = String.fromCharCode(...bytes);
        const existing = this.#ids.get(key);
        if (existing !== undefined) return existing;
        if (this.#tokens.length >= this.#maxTokens) return null;
        const id = this.#tokens.length;
        this.#tokens.push(bytes);
        this.#ids.set(key, id);
        let node = this.#root;
        for (const byte of bytes) {
            let child = node.children.get(byte);
            if (!child) {
                child = { children: new Map() };
                node.children.set(byte, child);
            }
            node = child;
        }
        node.token = id;
        return id;
    }

    encode(text: string): number[] {
        const bytes = new TextEncoder().encode(text);
        const tokens: number[] = [];
        let position = 0;
        while (position < bytes.length) {
            let node = this.#root;
            let matched = -1;
            let matchedEnd = position;
            let index = position;
            let searching = true;
            while (index < bytes.length && searching) {
                const child = node.children.get(bytes[index] as number);
                if (!child) {
                    searching = false;
                } else {
                    node = child;
                    if (node.token !== undefined) {
                        matched = node.token;
                        matchedEnd = index + 1;
                    }
                    index += 1;
                }
            }
            if (matched >= 0) {
                tokens.push(matched);
                position = matchedEnd;
            } else {
                tokens.push(bytes[position] as number);
                position += 1;
            }
        }
        return tokens;
    }

    decode(tokens: readonly number[]): string {
        const bytes = new Uint8Array(
            tokens.reduce((total, token) => {
                if (!Number.isInteger(token) || token < 0 || token >= this.#tokens.length) {
                    throw new RangeError(`token fuera de vocabulario: ${token}`);
                }
                return total + (this.#tokens[token] as Uint8Array).length;
            }, 0),
        );
        let offset = 0;
        for (const token of tokens) {
            const bytesToken = this.#tokens[token] as Uint8Array;
            bytes.set(bytesToken, offset);
            offset += bytesToken.length;
        }
        return new TextDecoder().decode(bytes);
    }
}
