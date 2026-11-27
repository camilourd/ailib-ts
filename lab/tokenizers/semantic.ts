// F9.3b: semántica temporal y multiobjetivo tokens vs BPB proxy.
// Compara base, separación de terminaciones verbales y etiquetas de tipo.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const externalAll = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, 20000);
const validation = quijote.slice(20000, 28000);
const external = externalAll.slice(0, 10000);
const UNIT = /\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu;
const bytes = (value: string): number => new TextEncoder().encode(value).length;

const endings = [
    "ábamos", "ábais", "abais", "aban", "abas", "aba",
    "íamos", "íais", "ían", "ías", "ía",
    "ieron", "ieron", "ieron", "ió", "aste", "aron", "ado", "ido", "ando", "iendo",
    "aremos", "aréis", "arán", "ará", "eremos", "eréis", "erán", "erá",
    "iremos", "iréis", "irán", "irá", "aría", "arías", "arían",
    "amos", "áis", "an", "as", "en", "es", "emos", "éis",
].sort((first, second) => second.length - first.length);

const counts = new Map<string, number>();
const unitList = train.match(UNIT) ?? [];
for (const unit of unitList) {
    const characters = [...unit];
    for (let start = 0; start < characters.length; start++) {
        let candidate = "";
        for (
            let end = start;
            end < characters.length && bytes(candidate + characters[end]) <= 64;
            end++
        ) {
            candidate += characters[end] as string;
            counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
        }
    }
}
for (let index = 0; index < unitList.length; index++) {
    let candidate = "";
    for (
        let next = index;
        next < unitList.length && next < index + 4 && bytes(candidate + unitList[next]) <= 64;
        next++
    ) {
        candidate += unitList[next] as string;
        counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
    }
}
const baseTokens: Uint8Array[] = [];
for (let byte = 0; byte < 256; byte++) baseTokens.push(Uint8Array.from([byte]));
for (const [value] of [...counts.entries()]
    .filter(([value, count]) => count >= 2 && bytes(value) <= 64)
    .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
    .slice(0, 4096 - 256)) {
    baseTokens.push(new TextEncoder().encode(value));
}

class SemanticTokenizer {
    readonly #base: LosslessByteTokenizer;
    readonly #mode: "none" | "suffix" | "tag";
    readonly #suffixes: string[] = endings;
    readonly #suffixIds = new Map<string, number>();
    readonly #tags = ["<WORD>", "<NUM>", "<PUNCT>", "<SPACE>"];

    constructor(mode: "none" | "suffix" | "tag") {
        this.#mode = mode;
        this.#base = new LosslessByteTokenizer(4096);
        for (const token of baseTokens.slice(256)) this.#base.add(token);
        if (mode === "suffix") {
            let id = this.#base.vocabularySize;
            for (const suffix of this.#suffixes) {
                this.#suffixIds.set(suffix, id);
                id += 1;
            }
        }
    }

    get vocabularySize(): number {
        if (this.#mode === "suffix") return this.#base.vocabularySize + this.#suffixes.length;
        if (this.#mode === "tag") return this.#base.vocabularySize + this.#tags.length;
        return this.#base.vocabularySize;
    }

    encode(text: string): number[] {
        if (this.#mode === "none") return this.#base.encode(text);
        if (this.#mode === "tag") {
            const tokens = this.#base.encode(text);
            const base = this.#base.vocabularySize;
            const output: number[] = [];
            for (const token of tokens) {
                const first = this.#base.decode([token])[0] ?? " ";
                const code = first.charCodeAt(0);
                const tag = code === 32 || code === 9 || code === 10
                    ? 3
                    : code >= 48 && code <= 57
                      ? 1
                      : (code >= 65 && code <= 90) || (code >= 97 && code <= 122) || code >= 128
                        ? 0
                        : 2;
                output.push(base + tag, token);
            }
            return output;
        }
        const output: number[] = [];
        for (const unit of text.match(UNIT) ?? []) {
            if (!/^\p{L}+$/u.test(unit)) {
                output.push(...this.#base.encode(unit));
            } else {
                const lower = unit.toLocaleLowerCase();
                const suffix = unit === lower
                    ? this.#suffixes.find(
                          (candidate) => lower.endsWith(candidate) && unit.length > candidate.length,
                      )
                    : undefined;
                if (!suffix) {
                    output.push(...this.#base.encode(unit));
                } else {
                    const stem = unit.slice(0, unit.length - suffix.length);
                    output.push(...this.#base.encode(stem), this.#suffixIds.get(suffix) as number);
                }
            }
        }
        return output;
    }

    decode(tokens: readonly number[]): string {
        if (this.#mode === "none") return this.#base.decode(tokens);
        if (this.#mode === "tag") {
            const base = this.#base.vocabularySize;
            return this.#base.decode(tokens.filter((token) => token < base));
        }
        const base = this.#base.vocabularySize;
        let text = "";
        let group: number[] = [];
        for (const token of tokens) {
            if (token < base) {
                group.push(token);
            } else {
                if (group.length > 0) {
                    text += this.#base.decode(group);
                    group = [];
                }
                text += this.#suffixes[token - base] ?? "";
            }
        }
        if (group.length > 0) text += this.#base.decode(group);
        return text;
    }
}

function evaluate(tokenizer: SemanticTokenizer): { tokens: number; external: number; exact: boolean } {
    const valTokens = tokenizer.encode(validation);
    const extTokens = tokenizer.encode(external);
    return {
        tokens: valTokens.length / [...validation].length,
        external: extTokens.length / [...external].length,
        exact:
            tokenizer.decode(valTokens) === validation &&
            tokenizer.decode(extTokens) === external,
    };
}

function trainProxy(tokenizer: SemanticTokenizer): { bpb: number; externalBpb: number; parameters: number } {
    const model = new TransformerLM(Math.max(256, tokenizer.vocabularySize), {
        modelSize: 32,
        heads: 4,
        blocks: 1,
        hiddenSize: 64,
        seed: 1,
    });
    const optimizer = new SGD(model.parameters(), 0.005);
    const trainTokens = tokenizer.encode(train);
    const rng = createRng(1);
    const context = 64;
    for (let step = 0; step < 300; step++) {
        const start = rng.int(trainTokens.length - context - 1);
        optimizer.zeroGrad();
        const loss = model.loss(trainTokens.slice(start, start + context + 1));
        loss.backward();
        clipGradients(model.parameters(), 1);
        optimizer.step();
    }
    const measure = (text: string): number => {
        const tokens = tokenizer.encode(text);
        const stride = Math.max(1, Math.floor((tokens.length - context - 1) / 4));
        let total = 0;
        for (let window = 0; window < 4; window++) {
            const start = window * stride;
            total += model.loss(tokens.slice(start, start + context + 1)).get();
        }
        return total / 4;
    };
    return {
        bpb: (measure(validation) * tokenizer.encode(validation).length) / Buffer.byteLength(validation, "utf8") / Math.LN2,
        externalBpb:
            (measure(external) * tokenizer.encode(external).length) /
            Buffer.byteLength(external, "utf8") /
            Math.LN2,
        parameters: model.parameterCount(),
    };
}

const rows = [];
for (const mode of ["none", "suffix", "tag"] as const) {
    const tokenizer = new SemanticTokenizer(mode);
    const tokens = evaluate(tokenizer);
    if (!tokens.exact) {
        const encoded = tokenizer.encode(validation);
        const decoded = tokenizer.decode(encoded);
        let mismatch = 0;
        while (mismatch < validation.length && validation[mismatch] === decoded[mismatch]) {
            mismatch += 1;
        }
        console.log(
            `  fallo ${mode} en ${mismatch}: esperado «${validation.slice(mismatch, mismatch + 20)}» ` +
                `obtenido «${decoded.slice(mismatch, mismatch + 20)}»`,
        );
    }
    const proxy = trainProxy(tokenizer);
    rows.push({ mode, ...tokens, ...proxy });
    console.log(
        `${mode.padEnd(6)} vocab ${tokenizer.vocabularySize} · val ${tokens.tokens.toFixed(3)} · ` +
            `externo ${tokens.external.toFixed(3)} · BPB ${proxy.bpb.toFixed(3)} · ` +
            `BPB ext ${proxy.externalBpb.toFixed(3)} · exacto ${tokens.exact ? "sí" : "no"}`,
    );
}
writeFileSync(
    "data/tokenizer-semantic.json",
    `${JSON.stringify({ train: [...train].length, validation: [...validation].length, external: [...external].length, rows }, null, 2)}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-semantic.json");
