// F9.3c: barrido v2 con cuatro operadores sobre el genoma de tokenizers.
// Genoma: fuente, vocabulario, maxLen, frecuencia y política. Fitness barato (tokens).

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
const bytes = (value: string): number => new TextEncoder().encode(value).length;

const wordCounts = new Map<string, number>();
const substringCounts = new Map<string, number>();
const charCounts = new Map<string, number>();
const unitList = train.match(UNIT) ?? [];
for (const unit of unitList) {
    wordCounts.set(unit, (wordCounts.get(unit) ?? 0) + 1);
    for (const character of unit) charCounts.set(character, (charCounts.get(character) ?? 0) + 1);
    const characters = [...unit];
    for (let start = 0; start < characters.length; start++) {
        let candidate = "";
        for (
            let end = start;
            end < characters.length && bytes(candidate + characters[end]) <= 128;
            end++
        ) {
            candidate += characters[end] as string;
            substringCounts.set(candidate, (substringCounts.get(candidate) ?? 0) + 1);
        }
    }
}
const bothCounts = new Map(wordCounts);
for (const [value, count] of substringCounts) {
    bothCounts.set(value, (bothCounts.get(value) ?? 0) + count);
}
const SOURCES = ["word", "substring", "both", "char"] as const;
const SOURCE_COUNTS = [wordCounts, substringCounts, bothCounts, charCounts];
const VOCABULARIES = [2048, 4096, 8192, 16384, 32768];
const LENGTHS = [32, 64, 128];
const FREQUENCIES = [1, 2, 5, 10];
const POLICIES = ["frecuencia", "ganancia"] as const;

type Genome = number[];
const limits = [SOURCES.length, VOCABULARIES.length, LENGTHS.length, FREQUENCIES.length, POLICIES.length];

interface Evaluation {
    genome: Genome;
    source: string;
    vocabulary: number;
    maxLen: number;
    minFrequency: number;
    policy: string;
    validation: number;
    external: number;
    score: number;
    exact: boolean;
}

const cache = new Map<string, Evaluation>();
let builds = 0;

function evaluate(genome: Genome): Evaluation {
    const key = genome.join(",");
    const cached = cache.get(key);
    if (cached) return cached;
    builds += 1;
    const source = SOURCES[(genome[0] as number) % SOURCES.length] as string;
    const vocabulary = VOCABULARIES[(genome[1] as number) % VOCABULARIES.length] as number;
    const maxLen = LENGTHS[(genome[2] as number) % LENGTHS.length] as number;
    const minFrequency = FREQUENCIES[(genome[3] as number) % FREQUENCIES.length] as number;
    const policy = POLICIES[(genome[4] as number) % POLICIES.length] as string;
    const counts = SOURCE_COUNTS[(genome[0] as number) % SOURCE_COUNTS.length] as Map<string, number>;
    const candidates = [...counts.entries()]
        .filter(([value, count]) => count >= minFrequency && bytes(value) <= maxLen)
        .map(([value, count]) => ({
            value,
            score: policy === "frecuencia" ? count : count * bytes(value),
        }))
        .sort((first, second) => second.score - first.score || first.value.localeCompare(second.value));
    const tokenizer = new LosslessByteTokenizer(vocabulary);
    for (const candidate of candidates.slice(0, vocabulary - 256)) {
        tokenizer.add(new TextEncoder().encode(candidate.value));
    }
    const exact = isLossless(tokenizer, [train, validation, external]);
    const validationRatio = tokenizer.encode(validation).length / [...validation].length;
    const externalRatio = tokenizer.encode(external).length / [...external].length;
    const score = validationRatio + 0.5 * externalRatio;
    const result = {
        genome: [...genome],
        source,
        vocabulary,
        maxLen,
        minFrequency,
        policy,
        validation: validationRatio,
        external: externalRatio,
        score,
        exact,
    };
    cache.set(key, result);
    return result;
}

function randomGenome(rng: ReturnType<typeof createRng>): Genome {
    return limits.map((limit) => rng.int(limit));
}

const allCombinations: Genome[] = [];
for (const source of limits[0] ? Array.from({ length: SOURCES.length }, (_, index) => index) : []) {
    for (let vocabulary = 0; vocabulary < VOCABULARIES.length; vocabulary++) {
        for (let length = 0; length < LENGTHS.length; length++) {
            for (let frequency = 0; frequency < FREQUENCIES.length; frequency++) {
                for (let policy = 0; policy < POLICIES.length; policy++) {
                    allCombinations.push([source, vocabulary, length, frequency, policy]);
                }
            }
        }
    }
}
const exhaustive = allCombinations.map(evaluate);
const randomBest = exhaustive.reduce((best, current) => (current.score < best.score ? current : best));

interface Run {
    name: string;
    best: Evaluation;
    history: number[];
}

function runAlgorithm(
    name: string,
    seed: number,
    next: (scored: Evaluation[], rng: ReturnType<typeof createRng>) => Genome[],
): Run {
    const rng = createRng(seed);
    let population = Array.from({ length: 100 }, () => randomGenome(rng));
    let best: Evaluation | null = null;
    const history: number[] = [];
    for (let generation = 0; generation < 100; generation++) {
        const scored = population.map(evaluate).sort((first, second) => first.score - second.score);
        if (!best || (scored[0] as Evaluation).score < best.score) best = scored[0] as Evaluation;
        history.push(best.score);
        population = next(scored, rng);
    }
    return { name, best: best as Evaluation, history };
}

function tournament(scored: Evaluation[], rng: ReturnType<typeof createRng>): Genome {
    const first = scored[rng.int(scored.length)] as Evaluation;
    const second = scored[rng.int(scored.length)] as Evaluation;
    return (first.score <= second.score ? first : second).genome;
}

function mutate(child: Genome, rng: ReturnType<typeof createRng>, rate: number): void {
    for (let index = 0; index < child.length; index++) {
        if (rng.next() < rate) {
            child[index] = (child[index] as number) + 1 + rng.int((limits[index] as number) - 1);
            child[index] = (child[index] as number) % (limits[index] as number);
        }
    }
}

const methods = [
    runAlgorithm("SGE", 2, (scored, rng) => {
        const next = scored.slice(0, 4).map((result) => result.genome);
        while (next.length < 100) {
            const first = tournament(scored, rng);
            const second = tournament(scored, rng);
            const child = first.map((gene, index) => (rng.next() < 0.5 ? gene : (second[index] as number)));
            mutate(child, rng, 0.2);
            next.push(child);
        }
        return next;
    }),
    runAlgorithm("GE", 3, (scored, rng) => {
        const next = scored.slice(0, 4).map((result) => result.genome);
        while (next.length < 100) {
            const first = tournament(scored, rng);
            const second = tournament(scored, rng);
            const cut = 1 + rng.int(first.length - 1);
            const child = [...first.slice(0, cut), ...second.slice(cut)];
            mutate(child, rng, 0.3);
            next.push(child);
        }
        return next;
    }),
    runAlgorithm("πGE", 4, (scored, rng) => {
        const next = scored.slice(0, 4).map((result) => result.genome);
        while (next.length < 100) {
            const first = tournament(scored, rng);
            const second = tournament(scored, rng);
            const order = [0, 1, 2, 3, 4].sort(() => rng.next() - 0.5);
            const child = [...first];
            for (const index of order) {
                if (rng.next() < 0.5) child[index] = second[index] as number;
            }
            mutate(child, rng, 0.15);
            next.push(child);
        }
        return next;
    }),
    runAlgorithm("HAEA", 5, (scored, rng) => {
        const next = scored.slice(0, 4).map((result) => result.genome);
        while (next.length < 100) {
            const parent = tournament(scored, rng);
            const child = [...parent];
            mutate(child, rng, 0.1 + rng.next() * 0.4);
            next.push(child);
        }
        return next;
    }),
];

console.log(`v2: ${allCombinations.length} combinaciones · ${builds} tokenizers · ${cache.size} únicos`);
console.log(
    `aleatorio: ${randomBest.source}/${randomBest.vocabulary}/${randomBest.maxLen}/${randomBest.minFrequency}/${randomBest.policy} · ` +
        `val ${randomBest.validation.toFixed(3)} · ext ${randomBest.external.toFixed(3)}`,
);
for (const method of methods) {
    console.log(
        `${method.name.padEnd(5)}: ${method.best.source}/${method.best.vocabulary}/${method.best.maxLen}/${method.best.minFrequency}/${method.best.policy} · ` +
            `val ${method.best.validation.toFixed(3)} · ext ${method.best.external.toFixed(3)} · score ${method.best.score.toFixed(3)}`,
    );
}

const byVocabulary = VOCABULARIES.map((vocabulary) => {
    const candidates = [...cache.values()].filter((result) => result.vocabulary === vocabulary);
    return candidates.reduce((best, current) => (current.score < best.score ? current : best));
});
for (const result of byVocabulary) {
    console.log(
        `  vocab ${result.vocabulary}: ${result.source}/${result.maxLen}/${result.minFrequency}/${result.policy} · ` +
            `val ${result.validation.toFixed(3)} · ext ${result.external.toFixed(3)}`,
    );
}

writeFileSync(
    "data/tokenizer-sweep-v2.json",
    `${JSON.stringify(
        {
            protocol: { population: 100, generations: 100, combinations: allCombinations.length, train: [...train].length, validation: [...validation].length, external: [...external].length },
            randomBest,
            methods: methods.map((method) => ({ name: method.name, best: method.best, history: method.history })),
            byVocabulary,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-sweep-v2.json");
