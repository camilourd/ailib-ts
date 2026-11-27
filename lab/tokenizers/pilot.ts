// F9.2: piloto evolutivo barato. Genoma de 4 genes, sin LM; fitness = tokens/carácter.
// Se comparan búsqueda aleatoria y una SGE mínima (genes por campo, cruce por gen).

import { readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import {
    buildTokenizer,
    isLossless,
    type TokenizerProgram,
} from "../../src/data/tokenizer-program.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const externalAll = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, 20000);
const validation = quijote.slice(20000, 28000);
const external = externalAll.slice(0, 10000);

const FAMILIES = ["char", "word", "dict"] as const;
const VOCABULARIES = [2048, 4096];
const LENGTHS = [32, 64];
const FREQUENCIES = [1, 2, 5];

type Genome = number[];

function makeProgram(genome: Genome): TokenizerProgram {
    return {
        family: FAMILIES[(genome[0] as number) % FAMILIES.length] as TokenizerProgram["family"],
        vocabularySize: VOCABULARIES[(genome[1] as number) % VOCABULARIES.length] as number,
        maxTokenBytes: LENGTHS[(genome[2] as number) % LENGTHS.length] as number,
        minFrequency: FREQUENCIES[(genome[3] as number) % FREQUENCIES.length] as number,
    };
}

interface Evaluation {
    genome: Genome;
    family: string;
    vocabularySize: number;
    maxTokenBytes: number;
    minFrequency: number;
    validation: number;
    external: number;
    usage: number;
    score: number;
    exact: boolean;
}

const cache = new Map<string, Evaluation>();
let evaluations = 0;

function evaluate(genome: Genome): Evaluation {
    const key = genome.join(",");
    const cached = cache.get(key);
    if (cached) return cached;
    evaluations += 1;
    const program = makeProgram(genome);
    const tokenizer = buildTokenizer(program, train);
    const exact = isLossless(tokenizer, [train, validation, external]);
    const validationRatio = tokenizer.encode(validation).length / [...validation].length;
    const externalRatio = tokenizer.encode(external).length / [...external].length;
    const used = new Set(tokenizer.encode(train)).size;
    const usage = used / tokenizer.vocabularySize;
    const score = exact
        ? validationRatio + 0.5 * externalRatio + 0.1 * (1 - usage)
        : Number.POSITIVE_INFINITY;
    const result: Evaluation = {
        genome: [...genome],
        family: program.family,
        vocabularySize: program.vocabularySize,
        maxTokenBytes: program.maxTokenBytes,
        minFrequency: program.minFrequency,
        validation: validationRatio,
        external: externalRatio,
        usage,
        score,
        exact,
    };
    cache.set(key, result);
    return result;
}

function randomGenome(rng: ReturnType<typeof createRng>): Genome {
    return [rng.int(FAMILIES.length), rng.int(VOCABULARIES.length), rng.int(LENGTHS.length), rng.int(FREQUENCIES.length)];
}

const randomRng = createRng(1);
const randomHistory: number[] = [];
let randomBest: Evaluation | null = null;
for (let step = 0; step < 200; step++) {
    const result = evaluate(randomGenome(randomRng));
    if (!randomBest || result.score < randomBest.score) randomBest = result;
    randomHistory.push(randomBest.score);
}

const rng = createRng(2);
let population = Array.from({ length: 20 }, () => randomGenome(rng));
const populationHistory: number[] = [];
let best: Evaluation | null = null;
for (let generation = 0; generation < 10; generation++) {
    const scored = population.map(evaluate).sort((first, second) => first.score - second.score);
    if (!best || (scored[0] as Evaluation).score < best.score) best = scored[0] as Evaluation;
    populationHistory.push(best.score);
    const next: Genome[] = scored.slice(0, 2).map((result) => result.genome);
    while (next.length < 20) {
        const tournament = (): Genome => {
            const a = scored[rng.int(scored.length)] as Evaluation;
            const b = scored[rng.int(scored.length)] as Evaluation;
            return (a.score <= b.score ? a : b).genome;
        };
        const first = tournament();
        const second = tournament();
        const child = first.map((gene, index) =>
            rng.next() < 0.5 ? gene : (second[index] as number),
        );
        if (rng.next() < 0.3) {
            const index = rng.int(child.length);
            const limits = [FAMILIES.length, VOCABULARIES.length, LENGTHS.length, FREQUENCIES.length];
            child[index] = (child[index] as number) + 1 + rng.int((limits[index] as number) - 1);
            child[index] = (child[index] as number) % (limits[index] as number);
        }
        next.push(child);
    }
    population = next;
}

console.log(
    `piloto: ${evaluations} evaluaciones únicas · train ${[...train].length} car · ` +
        `val ${[...validation].length} · externo ${[...external].length}`,
);
if (randomBest) {
    console.log(
        `aleatorio: ${randomBest.family}/${randomBest.vocabularySize}/${randomBest.maxTokenBytes}/${randomBest.minFrequency} · ` +
            `val ${randomBest.validation.toFixed(3)} · externo ${randomBest.external.toFixed(3)} · score ${randomBest.score.toFixed(3)}`,
    );
}
if (best) {
    console.log(
        `SGE: ${best.family}/${best.vocabularySize}/${best.maxTokenBytes}/${best.minFrequency} · ` +
            `val ${best.validation.toFixed(3)} · externo ${best.external.toFixed(3)} · score ${best.score.toFixed(3)}`,
    );
}

writeFileSync(
    "data/tokenizer-pilot.json",
    `${JSON.stringify(
        {
            protocol: {
                population: 20,
                generations: 10,
                randomEvaluations: 200,
                trainCharacters: [...train].length,
                validationCharacters: [...validation].length,
                externalCharacters: [...external].length,
            },
            evaluations,
            randomBest,
            evolveBest: best,
            randomHistory,
            populationHistory,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-pilot.json");
