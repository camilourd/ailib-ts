// F9.3: barrido 100×100 con genoma de 5 genes y fitness barato.
// Familias word/substring/char, vocabulario 2k..32k, maxLen 32/64/128, frecuencia y política.

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";
import { isLossless } from "../../src/data/tokenizer-program.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const externalAll = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, 20000);
const validation = quijote.slice(20000, 28000);
const external = externalAll.slice(0, 10000);

const UNIT = /\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu;
const SOURCES = ["word", "substring", "char"] as const;
const VOCABULARIES = [2048, 4096, 8192, 16384, 32768];
const LENGTHS = [32, 64, 128];
const FREQUENCIES = [1, 2, 5, 10];
const POLICIES = ["frecuencia", "ganancia"] as const;
const bytes = (value: string): number => new TextEncoder().encode(value).length;

function units(text: string): string[] {
    return text.match(UNIT) ?? [];
}

const wordCounts = new Map<string, number>();
const substringCounts = new Map<string, number>();
const charCounts = new Map<string, number>();
for (const unit of units(train)) {
    wordCounts.set(unit, (wordCounts.get(unit) ?? 0) + 1);
    for (const character of unit) {
        charCounts.set(character, (charCounts.get(character) ?? 0) + 1);
    }
    const characters = [...unit];
    for (let start = 0; start < characters.length; start++) {
        let candidate = "";
        for (let end = start; end < characters.length; end++) {
            candidate += characters[end] as string;
            if (bytes(candidate) > 128) break;
            substringCounts.set(candidate, (substringCounts.get(candidate) ?? 0) + 1);
        }
    }
}

interface Evaluation {
    genome: number[];
    source: string;
    vocabularySize: number;
    maxTokenBytes: number;
    minFrequency: number;
    policy: string;
    validation: number;
    external: number;
    usage: number;
    score: number;
    exact: boolean;
}

const cache = new Map<string, Evaluation>();
let builds = 0;

function evaluate(genome: number[]): Evaluation {
    const key = genome.join(",");
    const cached = cache.get(key);
    if (cached) return cached;
    builds += 1;
    const source = SOURCES[(genome[0] as number) % SOURCES.length] as string;
    const vocabularySize = VOCABULARIES[(genome[1] as number) % VOCABULARIES.length] as number;
    const maxTokenBytes = LENGTHS[(genome[2] as number) % LENGTHS.length] as number;
    const minFrequency = FREQUENCIES[(genome[3] as number) % FREQUENCIES.length] as number;
    const policy = POLICIES[(genome[4] as number) % POLICIES.length] as string;
    const counts = source === "word" ? wordCounts : source === "char" ? charCounts : substringCounts;
    const candidates = [...counts.entries()]
        .filter(
            ([value, count]) =>
                count >= minFrequency && bytes(value) <= maxTokenBytes,
        )
        .map(([value, count]) => ({
            value,
            count,
            score: policy === "frecuencia" ? count : count * bytes(value),
        }))
        .sort((first, second) => second.score - first.score || first.value.localeCompare(second.value));
    const tokenizer = new LosslessByteTokenizer(vocabularySize);
    for (const candidate of candidates) {
        if (tokenizer.add(new TextEncoder().encode(candidate.value)) === null) break;
    }
    const exact = isLossless(tokenizer, [train, validation, external]);
    const validationRatio = tokenizer.encode(validation).length / [...validation].length;
    const externalRatio = tokenizer.encode(external).length / [...external].length;
    const usage = new Set(tokenizer.encode(train)).size / tokenizer.vocabularySize;
    const score = exact
        ? validationRatio + 0.5 * externalRatio + 0.1 * (1 - usage)
        : Number.POSITIVE_INFINITY;
    const result: Evaluation = {
        genome: [...genome],
        source,
        vocabularySize,
        maxTokenBytes,
        minFrequency,
        policy,
        validation: validationRatio,
        external: externalRatio,
        usage,
        score,
        exact,
    };
    cache.set(key, result);
    return result;
}

const limits = [
    SOURCES.length,
    VOCABULARIES.length,
    LENGTHS.length,
    FREQUENCIES.length,
    POLICIES.length,
];
function randomGenome(rng: ReturnType<typeof createRng>): number[] {
    return limits.map((limit) => rng.int(limit));
}

// Baseline: todo el espacio discreto (360 combinaciones).
const exhaustiveRng = createRng(1);
const all = new Set<string>();
for (let step = 0; step < 500; step++) {
    const genome = randomGenome(exhaustiveRng);
    all.add(genome.join(","));
    evaluate(genome);
}
const randomBest = [...all]
    .map((key) => cache.get(key) as Evaluation)
    .reduce((best, current) => (current.score < best.score ? current : best));

// SGE 100×100.
const rng = createRng(2);
let population = Array.from({ length: 100 }, () => randomGenome(rng));
let best: Evaluation | null = null;
const history: number[] = [];
for (let generation = 0; generation < 100; generation++) {
    const scored = population.map(evaluate).sort((first, second) => first.score - second.score);
    if (!best || (scored[0] as Evaluation).score < best.score) best = scored[0] as Evaluation;
    history.push(best.score);
    const next = scored.slice(0, 4).map((result) => result.genome);
    while (next.length < 100) {
        const tournament = (): number[] => {
            const first = scored[rng.int(scored.length)] as Evaluation;
            const second = scored[rng.int(scored.length)] as Evaluation;
            return (first.score <= second.score ? first : second).genome;
        };
        const first = tournament();
        const second = tournament();
        const child = first.map((gene, index) => (rng.next() < 0.5 ? gene : (second[index] as number)));
        for (let index = 0; index < child.length; index++) {
            if (rng.next() < 0.2) {
                child[index] = (child[index] as number) + 1 + rng.int((limits[index] as number) - 1);
                child[index] = (child[index] as number) % (limits[index] as number);
            }
        }
        next.push(child);
    }
    population = next;
    if (generation % 10 === 0) {
        console.log(
            `generación ${String(generation).padStart(3)} · best ${best.score.toFixed(3)} · ` +
                `${best.source}/${best.vocabularySize}/${best.maxTokenBytes}/${best.minFrequency}/${best.policy}`,
        );
    }
}

console.log(
    `barrido: ${builds} tokenizers construidos · ${cache.size} únicos · ` +
        `val ${[...validation].length} · externo ${[...external].length}`,
);
console.log(
    `aleatorio: ${randomBest.source}/${randomBest.vocabularySize}/${randomBest.maxTokenBytes}/${randomBest.minFrequency}/${randomBest.policy} · ` +
        `val ${randomBest.validation.toFixed(3)} · externo ${randomBest.external.toFixed(3)} · score ${randomBest.score.toFixed(3)}`,
);
console.log(
    `SGE 100×100: ${best?.source}/${best?.vocabularySize}/${best?.maxTokenBytes}/${best?.minFrequency}/${best?.policy} · ` +
        `val ${best?.validation.toFixed(3)} · externo ${best?.external.toFixed(3)} · score ${best?.score.toFixed(3)}`,
);

const unique = [...cache.values()];
const byVocabulary = VOCABULARIES.map((vocabulary) => {
    const candidates = unique.filter((result) => result.vocabularySize === vocabulary);
    return candidates.reduce((bestSoFar, current) =>
        current.score < bestSoFar.score ? current : bestSoFar,
    );
});
const bySource = SOURCES.map((source) => {
    const candidates = unique.filter((result) => result.source === source);
    return candidates.reduce((bestSoFar, current) =>
        current.score < bestSoFar.score ? current : bestSoFar,
    );
});
console.log("mejor por vocabulario:");
for (const result of byVocabulary) {
    console.log(
        `  ${result.vocabularySize}: ${result.source}/${result.maxTokenBytes}/${result.minFrequency}/${result.policy} · ` +
            `val ${result.validation.toFixed(3)} · externo ${result.external.toFixed(3)}`,
    );
}
console.log("mejor por fuente:");
for (const result of bySource) {
    console.log(
        `  ${result.source}: ${result.vocabularySize}/${result.maxTokenBytes}/${result.minFrequency}/${result.policy} · ` +
            `val ${result.validation.toFixed(3)} · externo ${result.external.toFixed(3)}`,
    );
}

writeFileSync(
    "data/tokenizer-sweep.json",
    `${JSON.stringify(
        {
            protocol: {
                population: 100,
                generations: 100,
                randomSamples: all.size,
                trainCharacters: [...train].length,
                validationCharacters: [...validation].length,
                externalCharacters: [...external].length,
                grid: { sources: SOURCES, vocabularies: VOCABULARIES, lengths: LENGTHS, frequencies: FREQUENCIES, policies: POLICIES },
            },
            builds,
            unique: cache.size,
            randomBest,
            sgeBest: best,
            byVocabulary,
            bySource,
            history,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-sweep.json");
