// Teoría: docs/teoria/evolucion/haea.md

import { createRng } from "../core/random/index.ts";
import type { RandomNumberGenerator } from "../core/random/index.ts";
import { tournament } from "./selection.ts";
import type { Fitness, Genome, Individual, Variation } from "./types.ts";

export interface HaeaOptions {
    populationSize: number;
    generations: number;
    operators: readonly Variation[];
    tournamentSize?: number;
    elitism?: number;
    seed?: number;
    adaptive?: boolean;
    init?: (rng: RandomNumberGenerator) => Genome;
    onGeneration?: (generation: number, best: Individual) => void;
}

export interface HaeaResult {
    best: Individual;
    history: number[];
    evaluations: number;
}

const cloneGenome = (genome: Genome): Genome => new Float32Array(genome);

function selectOperator(rates: readonly number[], rng: RandomNumberGenerator): number {
    let threshold = rng.next();
    for (let index = 0; index < rates.length; index++) {
        threshold -= rates[index] as number;
        if (threshold <= 0) return index;
    }
    return rates.length - 1;
}

function adjust(
    rates: number[],
    index: number,
    rng: RandomNumberGenerator,
    reward: boolean,
): void {
    const factor = reward ? 1 + rng.next() : 1 - rng.next();
    rates[index] = Math.max(1e-6, (rates[index] as number) * factor);
    let total = 0;
    for (const rate of rates) total += rate;
    for (let position = 0; position < rates.length; position++) {
        rates[position] = (rates[position] as number) / total;
    }
}

/**
 * HAEA simplificado: población real, torneo, operadores con probabilidades por individuo y
 * premio/castigo al operador elegido según mejore o no el descendiente.
 */
export function haea(fitness: Fitness, genomeSize: number, options: HaeaOptions): HaeaResult {
    if (!Number.isInteger(genomeSize) || genomeSize <= 0) {
        throw new RangeError(`el genoma debe tener un tamaño positivo, no ${genomeSize}`);
    }
    if (!Number.isInteger(options.populationSize) || options.populationSize < 2) {
        throw new RangeError("la población necesita al menos dos individuos");
    }
    if (!Number.isInteger(options.generations) || options.generations < 0) {
        throw new RangeError("las generaciones deben ser un entero no negativo");
    }
    if (options.operators.length === 0) throw new Error("hace falta al menos un operador");
    for (const operator of options.operators) {
        if (!Number.isInteger(operator.arity) || operator.arity < 1) {
            throw new Error(`el operador «${operator.name}» tiene aridad inválida`);
        }
    }
    const tournamentSize = options.tournamentSize ?? 3;
    if (!Number.isInteger(tournamentSize) || tournamentSize < 1) {
        throw new RangeError("el torneo debe tener al menos un individuo");
    }
    const elitism = options.elitism ?? 0;
    if (!Number.isInteger(elitism) || elitism < 0 || elitism > options.populationSize) {
        throw new RangeError("el elitismo debe caber en la población");
    }
    const adaptive = options.adaptive ?? true;
    const rng = createRng(options.seed ?? 1);
    const init =
        options.init ??
        ((random: RandomNumberGenerator): Genome => {
            const genome = new Float32Array(genomeSize);
            for (let index = 0; index < genomeSize; index++) {
                genome[index] = random.next() * 2 - 1;
            }
            return genome;
        });

    const population: Individual[] = [];
    let evaluations = 0;
    for (let index = 0; index < options.populationSize; index++) {
        const genome = init(rng);
        if (genome.length !== genomeSize) {
            throw new Error(`init devolvió un genoma de ${genome.length}, no ${genomeSize}`);
        }
        population.push({ genome, fitness: fitness(genome) });
        evaluations += 1;
    }

    const rates = population.map(() =>
        new Array<number>(options.operators.length).fill(1 / options.operators.length),
    );
    let best = population[0] as Individual;
    for (const individual of population) {
        if (individual.fitness < best.fitness) best = individual;
    }
    const history = [best.fitness];

    for (let generation = 0; generation < options.generations; generation++) {
        const elites =
            elitism > 0
                ? [...population]
                      .sort((left, right) => left.fitness - right.fitness)
                      .slice(0, elitism)
                      .map((individual) => ({
                          genome: cloneGenome(individual.genome),
                          fitness: individual.fitness,
                      }))
                : [];

        for (let index = 0; index < population.length; index++) {
            const individual = population[index] as Individual;
            const operatorIndex = selectOperator(rates[index] as number[], rng);
            const operator = options.operators[operatorIndex] as Variation;
            const parents: Genome[] = [individual.genome];
            for (let parent = 1; parent < operator.arity; parent++) {
                parents.push(tournament(population, tournamentSize, rng).genome);
            }
            const child = operator.apply(parents, rng);
            const childFitness = fitness(child);
            evaluations += 1;
            if (childFitness < individual.fitness) {
                population[index] = { genome: child, fitness: childFitness };
                if (adaptive) adjust(rates[index] as number[], operatorIndex, rng, true);
            } else if (adaptive) {
                adjust(rates[index] as number[], operatorIndex, rng, false);
            }
        }

        for (const elite of elites) {
            let worstIndex = 0;
            for (let index = 1; index < population.length; index++) {
                if (
                    (population[index] as Individual).fitness >
                    (population[worstIndex] as Individual).fitness
                ) {
                    worstIndex = index;
                }
            }
            if (elite.fitness < (population[worstIndex] as Individual).fitness) {
                population[worstIndex] = elite;
            }
        }

        let generationBest = population[0] as Individual;
        for (const individual of population) {
            if (individual.fitness < generationBest.fitness) generationBest = individual;
        }
        best = generationBest;
        history.push(best.fitness);
        options.onGeneration?.(generation, best);
    }

    return {
        best: { genome: cloneGenome(best.genome), fitness: best.fitness },
        history,
        evaluations,
    };
}
