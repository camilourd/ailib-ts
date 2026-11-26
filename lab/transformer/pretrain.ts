// Teoría: docs/teoria/entrenamiento/preentreno.md
// Preentreno del LM de F6: schedule, clipping, checkpoints y perplejidad.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { loadBPE } from "../../src/data/bpe-io.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { saveCheckpoint } from "../../src/io/checkpoint.ts";

const seed = Number(process.argv[2] ?? 1);
const steps = Number(process.argv[3] ?? 1200);
if (!Number.isInteger(seed) || seed < 0) throw new Error(`semilla inválida: ${seed}`);
if (!Number.isInteger(steps) || steps <= 0) throw new Error(`pasos inválidos: ${steps}`);

const context = 128;
const baseLearningRate = 0.003;
const warmup = 100;
const vocabularySize = 512;

const tokenizer = loadBPE("data/quijote-bpe.json");
const text = readFileSync("data/quijote.txt", "utf8");
const split = Math.floor(text.length * 0.9);
const trainTokens = tokenizer.encode(text.slice(0, split));
const validationTokens = tokenizer.encode(text.slice(split));

const model = new TransformerLM(vocabularySize, {
    modelSize: 160,
    heads: 8,
    blocks: 4,
    hiddenSize: 640,
    seed,
});
const optimizer = new SGD(model.parameters(), baseLearningRate);
const rng = createRng(seed + 1000);
mkdirSync("checkpoints", { recursive: true });

function learningRateAt(step: number): number {
    if (step < warmup) return (baseLearningRate * (step + 1)) / warmup;
    const progress = (step - warmup) / Math.max(1, steps - warmup);
    return Math.max(1e-5, baseLearningRate * 0.5 * (1 + Math.cos(Math.PI * progress)));
}

function evaluate(tokens: readonly number[], windows: number): number {
    const stride = Math.max(1, Math.floor((tokens.length - context - 1) / windows));
    let total = 0;
    for (let index = 0; index < windows; index++) {
        const start = index * stride;
        total += model.loss(tokens.slice(start, start + context + 1)).get();
    }
    return total / windows;
}

interface Point {
    step: number;
    learningRate: number;
    trainLoss: number;
    validationLoss: number;
    perplexity: number;
}

const history: Point[] = [];
const initialValidation = evaluate(validationTokens, 4);
console.log(
    `semilla ${seed} · ${model.parameterCount()} parámetros · ` +
        `${trainTokens.length} tokens train / ${validationTokens.length} val`,
);
console.log(`paso    0  lr ${baseLearningRate.toFixed(5)}  val ${initialValidation.toFixed(4)}  ppl ${Math.exp(initialValidation).toFixed(2)}`);

for (let step = 0; step < steps; step++) {
    const start = rng.int(trainTokens.length - context - 1);
    const piece = trainTokens.slice(start, start + context + 1);
    const lr = learningRateAt(step);
    optimizer.learningRate = lr;
    optimizer.zeroGrad();
    const loss = model.loss(piece);
    loss.backward();
    const gradientNorm = clipGradients(model.parameters(), 1);
    optimizer.step();
    const trainLoss = loss.get();

    if (step % 50 === 0 || step === steps - 1) {
        const validationLoss = evaluate(validationTokens, 4);
        const point: Point = {
            step,
            learningRate: lr,
            trainLoss,
            validationLoss,
            perplexity: Math.exp(validationLoss),
        };
        history.push(point);
        console.log(
            `paso ${String(step).padStart(4)}  lr ${lr.toFixed(5)}  ` +
                `train ${trainLoss.toFixed(4)}  val ${validationLoss.toFixed(4)}  ` +
                `ppl ${point.perplexity.toFixed(2)}  |g| ${gradientNorm.toFixed(2)}`,
        );
    }

    if (step > 0 && step % 200 === 0) {
        saveCheckpoint(`checkpoints/seed-${seed}-step-${step}.json`, model.namedParameters(), {
            seed,
            step,
            context,
            vocabularySize,
            architecture: model.architecture,
            parameterCount: model.parameterCount(),
            trainLoss,
            validationLoss: history.at(-1)?.validationLoss ?? initialValidation,
            perplexity: history.at(-1)?.perplexity ?? Math.exp(initialValidation),
        });
    }
}

const finalValidation = evaluate(validationTokens, 4);
const summary = {
    seed,
    steps,
    context,
    baseLearningRate,
    warmup,
    parameterCount: model.parameterCount(),
    architecture: model.architecture,
    initialValidationLoss: initialValidation,
    finalValidationLoss: finalValidation,
    finalPerplexity: Math.exp(finalValidation),
    history,
};
writeFileSync(`data/pretrain-seed-${seed}.json`, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(
    `final: val ${finalValidation.toFixed(4)} · ppl ${Math.exp(finalValidation).toFixed(2)} · ` +
        `data/pretrain-seed-${seed}.json`,
);

const start = tokenizer.encode("En un lugar de la Mancha");
const generated = model.generate(start, 80, { temperature: 0.8, seed: seed + 7 });
console.log("--- generado ---");
console.log(tokenizer.decode([...start, ...generated]));
