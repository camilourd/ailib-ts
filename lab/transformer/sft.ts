// Teoría: docs/teoria/entrenamiento/sft.md
// SFT: comparar el modelo preentrenado antes y después de ajustar con pares instrucción/respuesta.

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

import { createRng } from "../../src/core/random/index.ts";
import { loadBPE } from "../../src/data/bpe-io.ts";
import { loadCheckpoint, saveCheckpoint } from "../../src/io/checkpoint.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import { clipGradients } from "../../src/optim/clip.ts";
import { SGD } from "../../src/optim/sgd.ts";

interface Pair {
    instruction: string;
    response: string;
}

const dataset = JSON.parse(readFileSync("data/sft-quijote.json", "utf8")) as {
    train: Pair[];
    validation: Pair[];
};
const tokenizer = loadBPE("data/quijote-bpe.json");
const model = new TransformerLM(512, {
    modelSize: 160,
    heads: 8,
    blocks: 4,
    hiddenSize: 640,
    seed: 1,
});
const checkpoint = loadCheckpoint("checkpoints/seed-1-step-1000.json", { requiresGrad: true });
model.loadParameters(checkpoint.parameters);

const steps = 300;
const learningRate = 0.0005;
const optimizer = new SGD(model.parameters(), learningRate);
mkdirSync("checkpoints", { recursive: true });

const format = (pair: Pair): string =>
    `Pregunta: ${pair.instruction}\nRespuesta: ${pair.response}\n`;
const prompt = (pair: Pair): string => `Pregunta: ${pair.instruction}\nRespuesta: `;

function lossFor(pair: Pair): number {
    return model.loss(tokenizer.encode(format(pair))).get();
}

function evaluate(pairs: readonly Pair[]): { loss: number; perplexity: number } {
    let total = 0;
    for (const pair of pairs) total += lossFor(pair);
    const loss = total / pairs.length;
    return { loss, perplexity: Math.exp(loss) };
}

function answer(pair: Pair): string {
    const start = tokenizer.encode(prompt(pair));
    return tokenizer.decode(model.generate(start, 48, { temperature: 0.3, seed: 7 }));
}

const before = evaluate(dataset.validation);
const beforeSample = answer(dataset.validation[0] as Pair);
console.log(
    `antes: pérdida val ${before.loss.toFixed(4)} · ppl ${before.perplexity.toFixed(2)}`,
);
console.log(`antes: ${beforeSample.trim()}`);

interface Point {
    step: number;
    trainLoss: number;
    validationLoss: number;
}

const history: Point[] = [];
const rng = createRng(99);
for (let step = 0; step < steps; step++) {
    const pair = dataset.train[rng.int(dataset.train.length)] as Pair;
    optimizer.zeroGrad();
    const loss = model.loss(tokenizer.encode(format(pair)));
    loss.backward();
    clipGradients(model.parameters(), 1);
    optimizer.step();

    if (step % 50 === 0 || step === steps - 1) {
        const validation = evaluate(dataset.validation);
        const point = { step, trainLoss: loss.get(), validationLoss: validation.loss };
        history.push(point);
        console.log(
            `paso ${String(step).padStart(3)}  train ${point.trainLoss.toFixed(4)}  ` +
                `val ${validation.loss.toFixed(4)}`,
        );
    }
}

const after = evaluate(dataset.validation);
const afterSample = answer(dataset.validation[0] as Pair);
console.log(`después: pérdida val ${after.loss.toFixed(4)} · ppl ${after.perplexity.toFixed(2)}`);
console.log(`después: ${afterSample.trim()}`);

saveCheckpoint("checkpoints/sft-seed-1.json", model.namedParameters(), {
    stage: "sft",
    seed: 1,
    steps,
    learningRate,
    trainPairs: dataset.train.length,
    validationPairs: dataset.validation.length,
    before,
    after,
});
writeFileSync(
    "data/sft-report.json",
    `${JSON.stringify(
        {
            seed: 1,
            steps,
            learningRate,
            trainPairs: dataset.train.length,
            validationPairs: dataset.validation.length,
            before: { ...before, sample: beforeSample },
            after: { ...after, sample: afterSample },
            history,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/sft-report.json y checkpoints/sft-seed-1.json");
