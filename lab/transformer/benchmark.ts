// Teoría: docs/teoria/entrenamiento/inferencia.md
// Benchmark: generación sin caché, la línea base de F7.

import { writeFileSync } from "node:fs";

import { loadBPE } from "../../src/data/bpe-io.ts";
import { loadCheckpoint } from "../../src/io/checkpoint.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";

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

const prompt = tokenizer.encode("En un lugar de la Mancha");
const length = 48;
const runs = 3;
const temperature = 0.8;
const seed = 7;
const options = { temperature, seed };

const before = process.memoryUsage();
const warmupStart = performance.now();
model.generate(prompt, 8, options);
const warmupSeconds = (performance.now() - warmupStart) / 1000;

const samples: Array<{ seconds: number; tokensPerSecond: number }> = [];
let generated: number[] = [];
for (let run = 0; run < runs; run++) {
    const start = performance.now();
    generated = model.generate(prompt, length, options);
    const seconds = (performance.now() - start) / 1000;
    samples.push({ seconds, tokensPerSecond: length / seconds });
}
const after = process.memoryUsage();
const meanSeconds = samples.reduce((total, sample) => total + sample.seconds, 0) / samples.length;
const tokensPerSecond = length / meanSeconds;
const megabytes = (bytes: number): number => bytes / (1024 * 1024);

console.log(`modelo: ${model.parameterCount()} parámetros · checkpoint paso 1000`);
console.log(
    `prompt ${prompt.length} tokens · ${length} generados · ${runs} corridas · temp ${temperature} · semilla ${seed}`,
);
console.log(`warmup: ${warmupSeconds.toFixed(3)} s`);
for (const [index, sample] of samples.entries()) {
    console.log(
        `corrida ${index + 1}: ${(sample.seconds * 1000).toFixed(0)} ms · ` +
            `${sample.tokensPerSecond.toFixed(2)} tokens/s`,
    );
}
console.log(
    `media: ${(meanSeconds * 1000).toFixed(0)} ms · ${tokensPerSecond.toFixed(2)} tokens/s · ` +
        `heap ${megabytes(before.heapUsed).toFixed(1)} → ${megabytes(after.heapUsed).toFixed(1)} MB`,
);
console.log(`muestra: ${tokenizer.decode(generated).slice(0, 120)}`);

writeFileSync(
    "data/inference-baseline.json",
    `${JSON.stringify(
        {
            model: "transformer-1.4m",
            parameters: model.parameterCount(),
            checkpoint: "seed-1-step-1000",
            promptTokens: prompt.length,
            generatedTokens: length,
            temperature,
            seed,
            runs,
            warmupSeconds,
            samples,
            meanSeconds,
            tokensPerSecond,
            heapUsedBefore: before.heapUsed,
            heapUsedAfter: after.heapUsed,
            sample: tokenizer.decode(generated),
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/inference-baseline.json");
