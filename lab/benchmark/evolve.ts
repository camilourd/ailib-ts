// F10.3: evolución con genoma morfológico y fitness multidimensional.
// Nuevo genoma: modo morfológico, longitud mínima, presupuesto y longitud de sufijo.
// Nuevo fitness: tokens + frontera + ED, no solo tokens.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { BPETokenizer } from "../../src/data/bpe.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";

interface Reference {
    form: string;
    stem: string;
    suffix: string;
    tense: string;
    confidence: string;
}

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, Math.floor(quijote.length * 0.7));
const test = quijote.slice(Math.floor(quijote.length * 0.85));
const externalText = external;
const reference = JSON.parse(readFileSync("data/morphology-reference-reviewed.json", "utf8")) as {
    records: Reference[];
};

const endings = [
    "ábamos", "ábais", "abais", "aban", "abas", "aba",
    "íamos", "íais", "ían", "ías", "ía",
    "ieron", "ió", "aste", "aron", "ado", "ido", "ando", "iendo",
    "aremos", "aréis", "arán", "ará", "eremos", "eréis", "erán", "erá",
    "iremos", "iréis", "irán", "irá", "aría", "arías", "arían",
    "amos", "áis", "an", "as", "en", "es", "emos", "éis",
];
const prefixes = [
    "des", "re", "in", "im", "pre", "sub", "inter", "trans", "anti",
    "auto", "co", "ex", "extra", "hiper", "micro", "multi", "post",
    "super", "ultra",
];

const bpe = new BPETokenizer();
bpe.train(train, { vocabularySize: 2048, minFrequency: 2 });
const baseTokens: Uint8Array[] = [];
const seen = new Set<string>();
for (let byte = 0; byte < 256; byte++) {
    baseTokens.push(Uint8Array.from([byte]));
    seen.add(String.fromCharCode(byte));
}
for (let id = 0; id < bpe.vocabularySize; id++) {
    const text = bpe.decode([id]);
    if (!text.includes("\uFFFD")) {
        const bytes = new TextEncoder().encode(text);
        const key = String.fromCharCode(...bytes);
        if (bytes.length > 0 && !seen.has(key)) {
            seen.add(key);
            baseTokens.push(bytes);
        }
    }
}
const numberCounts = new Map<string, number>();
for (const match of train.matchAll(/\p{N}+/gu)) {
    numberCounts.set(match[0], (numberCounts.get(match[0]) ?? 0) + 1);
}
const numberTokens = [...numberCounts.entries()]
    .filter(([, count]) => count >= 2)
    .sort((first, second) => second[1] - first[1])
    .slice(0, 64)
    .map(([value]) => new TextEncoder().encode(value));

const MODES = ["ninguno", "sufijo", "prefijo+sufijo", "prefijo+sufijo+numero"] as const;
const MIN_LENGTHS = [1, 2, 3, 4, 5, 6];
const BUDGETS = [512, 768, 1024, 1280, 1536, 1792, 2048];
const MIN_SUFFIX = [2, 3, 4, 5];

type Genome = number[];
const limits = [MODES.length, MIN_LENGTHS.length, BUDGETS.length, MIN_SUFFIX.length];

interface Evaluation {
    genome: Genome;
    mode: string;
    tokensPerChar: number;
    boundaryF1: number;
    editDistance: number;
    score: number;
    vocabulary: number;
    exact: boolean;
}

const cache = new Map<string, Evaluation>();
let builds = 0;

function decode(token: Uint8Array): string {
    return new TextDecoder().decode(token);
}

function build(genome: Genome): { tokenizer: LosslessByteTokenizer; mode: string } {
    const mode = MODES[(genome[0] as number) % MODES.length] as string;
    const minLength = MIN_LENGTHS[(genome[1] as number) % MIN_LENGTHS.length] as number;
    const budget = BUDGETS[(genome[2] as number) % BUDGETS.length] as number;
    const minSuffix = MIN_SUFFIX[(genome[3] as number) % MIN_SUFFIX.length] as number;
    const extras: Uint8Array[] = [];
    if (mode !== "ninguno") {
        for (const suffix of endings) {
            const bytes = new TextEncoder().encode(suffix);
            if (suffix.length >= minSuffix && !seen.has(String.fromCharCode(...bytes))) extras.push(bytes);
        }
    }
    if (mode === "prefijo+sufijo" || mode === "prefijo+sufijo+numero") {
        for (const prefix of prefixes) {
            const bytes = new TextEncoder().encode(prefix);
            if (prefix.length >= minLength && !seen.has(String.fromCharCode(...bytes))) extras.push(bytes);
        }
    }
    if (mode === "prefijo+sufijo+numero") {
        extras.push(...numberTokens);
    }
    const tokenizer = new LosslessByteTokenizer(budget + extras.length);
    for (const token of baseTokens) {
        if (decode(token).length >= minLength) tokenizer.add(token);
    }
    for (const token of extras) tokenizer.add(token);
    return { tokenizer, mode };
}

function boundary(tokenizer: LosslessByteTokenizer): number {
    let truePositives = 0;
    let predictedTotal = 0;
    let support = 0;
    for (const record of reference.records) {
        if (record.confidence === "alta" && record.stem && record.suffix && record.stem.length < record.form.length) {
            support += 1;
            const pieces = tokenizer.encode(record.form).map((token) => tokenizer.decode([token]));
            const boundaries = new Set<number>();
            let offset = 0;
            for (let index = 0; index < pieces.length - 1; index++) {
                offset += (pieces[index] as string).length;
                boundaries.add(offset);
            }
            predictedTotal += boundaries.size;
            if (boundaries.has(record.stem.length)) truePositives += 1;
        }
    }
    const precision = predictedTotal === 0 ? 0 : truePositives / predictedTotal;
    const recall = support === 0 ? 0 : truePositives / support;
    return precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
}

function editDistance(first: string[], second: string[]): number {
    const previous = Array.from({ length: second.length + 1 }, (_, index) => index);
    for (let row = 1; row <= first.length; row++) {
        let diagonal = previous[0] as number;
        previous[0] = row;
        for (let column = 1; column <= second.length; column++) {
            const above = previous[column] as number;
            previous[column] = Math.min(
                (previous[column] as number) + 1,
                (previous[column - 1] as number) + 1,
                diagonal + (first[row - 1] === second[column - 1] ? 0 : 1),
            );
            diagonal = above;
        }
    }
    return previous[second.length] as number;
}

function evaluate(genome: Genome): Evaluation {
    const key = genome.join(",");
    const cached = cache.get(key);
    if (cached) return cached;
    builds += 1;
    const { tokenizer, mode } = build(genome);
    const testTokens = tokenizer.encode(test);
    const exact = tokenizer.decode(testTokens) === test;
    let distance = 0;
    let support = 0;
    for (const record of reference.records) {
        if (record.confidence === "alta" && record.stem && record.suffix) {
            const predicted = tokenizer.encode(record.form).map((token) => tokenizer.decode([token]));
            distance += editDistance([record.stem, record.suffix], predicted) /
                Math.max(2, predicted.length);
            support += 1;
        }
    }
    const meanDistance = support === 0 ? 0 : distance / support;
    const boundaryF1 = boundary(tokenizer);
    const tokensPerChar = testTokens.length / [...test].length;
    const score = tokensPerChar + 0.5 * (1 - boundaryF1) + 0.1 * meanDistance;
    const evaluation = {
        genome: [...genome],
        mode,
        tokensPerChar,
        boundaryF1,
        editDistance: meanDistance,
        score,
        vocabulary: tokenizer.vocabularySize,
        exact,
    };
    cache.set(key, evaluation);
    return evaluation;
}

const all: Genome[] = [];
for (let mode = 0; mode < MODES.length; mode++) {
    for (let length = 0; length < MIN_LENGTHS.length; length++) {
        for (let budget = 0; budget < BUDGETS.length; budget++) {
            for (let suffix = 0; suffix < MIN_SUFFIX.length; suffix++) {
                all.push([mode, length, budget, suffix]);
            }
        }
    }
}
const exhaustive = all.map(evaluate);
const randomBest = exhaustive.reduce((best, current) => (current.score < best.score ? current : best));
const base = evaluate([0, 0, BUDGETS.length - 1, 0]);
const suffix = evaluate([1, 0, BUDGETS.length - 1, 0]);
for (const [label, result] of [["base", base], ["sufijo", suffix]] as const) {
    console.log(
        `${label}: ${result.mode}/${result.vocabulary} · tokens ${result.tokensPerChar.toFixed(3)} · F1 ${result.boundaryF1.toFixed(3)} · ` +
            `ED ${result.editDistance.toFixed(3)} · score ${result.score.toFixed(3)}`,
    );
}

function run(seed: number) {
    const rng = createRng(seed);
    let population = Array.from({ length: 100 }, () => limits.map((limit) => rng.int(limit)));
    let best = randomBest;
    for (let generation = 0; generation < 100; generation++) {
        const scored = population.map(evaluate).sort((first, second) => first.score - second.score);
        if ((scored[0] as Evaluation).score < best.score) best = scored[0] as Evaluation;
        const next = scored.slice(0, 4).map((result) => result.genome);
        while (next.length < 100) {
            const first = scored[rng.int(scored.length)] as Evaluation;
            const second = scored[rng.int(scored.length)] as Evaluation;
            const child = first.genome.map((gene, index) =>
                rng.next() < 0.5 ? gene : (second.genome[index] as number),
            );
            for (let index = 0; index < child.length; index++) {
                if (rng.next() < 0.2) child[index] = rng.int(limits[index] as number);
            }
            next.push(child);
        }
        population = next;
    }
    return best;
}

const sge = run(2);
console.log(`combinaciones ${all.length} · tokenizers ${builds} · únicos ${cache.size}`);
console.log(
    `aleatorio: ${randomBest.mode}/${randomBest.vocabulary} · tokens ${randomBest.tokensPerChar.toFixed(3)} · ` +
        `F1 ${randomBest.boundaryF1.toFixed(3)} · ED ${randomBest.editDistance.toFixed(3)} · score ${randomBest.score.toFixed(3)}`,
);
console.log(
    `SGE:       ${sge.mode}/${sge.vocabulary} · tokens ${sge.tokensPerChar.toFixed(3)} · ` +
        `F1 ${sge.boundaryF1.toFixed(3)} · ED ${sge.editDistance.toFixed(3)} · score ${sge.score.toFixed(3)}`,
);

writeFileSync(
    "data/tokenizer-evolution-f10.json",
    `${JSON.stringify({ protocol: { population: 100, generations: 100, combinations: all.length, train: [...train].length, test: [...test].length, external: [...externalText].length }, randomBest, sge }, null, 2)}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-evolution-f10.json");
