// Teoría: docs/teoria/entrenamiento/inferencia.md
// Comparación: generación sin caché frente a KV cache, misma entrada y calidad.

import { readFileSync, writeFileSync } from "node:fs";

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
const options = { temperature: 0.8, seed: 7 };
const expected = model.generate(prompt, length, options);
const cachedOutput = model.generate(prompt, length, { ...options, useCache: true });
if (cachedOutput.join(",") !== expected.join(",")) {
    throw new Error("el KV cache cambió la generación");
}

function measure(useCache: boolean): { seconds: number; tokensPerSecond: number }[] {
    const samples: { seconds: number; tokensPerSecond: number }[] = [];
    for (let run = 0; run < runs; run++) {
        const start = performance.now();
        model.generate(prompt, length, { ...options, useCache });
        const seconds = (performance.now() - start) / 1000;
        samples.push({ seconds, tokensPerSecond: length / seconds });
    }
    return samples;
}

model.generate(prompt, 8, options);
const normal = measure(false);
model.generate(prompt, 8, { ...options, useCache: true });
const cached = measure(true);
const mean = (samples: { seconds: number }[]): number =>
    samples.reduce((total, sample) => total + sample.seconds, 0) / samples.length;
const normalSeconds = mean(normal);
const cachedSeconds = mean(cached);
const normalRate = length / normalSeconds;
const cachedRate = length / cachedSeconds;

console.log(`modelo: ${model.parameterCount()} parámetros · ${length} tokens · ${runs} corridas`);
console.log("sin caché:");
for (const sample of normal) {
    console.log(`  ${(sample.seconds * 1000).toFixed(0)} ms · ${sample.tokensPerSecond.toFixed(2)} tokens/s`);
}
console.log("con KV cache:");
for (const sample of cached) {
    console.log(`  ${(sample.seconds * 1000).toFixed(0)} ms · ${sample.tokensPerSecond.toFixed(2)} tokens/s`);
}
console.log(
    `media sin caché ${normalRate.toFixed(2)} tokens/s · con caché ${cachedRate.toFixed(2)} tokens/s · ` +
        `mejora ${(cachedRate / normalRate).toFixed(2)}×`,
);
console.log(`salida idéntica: ${cachedOutput.join(",") === expected.join(",")}`);

writeFileSync(
    "data/inference-kv-cache.json",
    `${JSON.stringify(
        {
            model: "transformer-1.4m",
            parameters: model.parameterCount(),
            checkpoint: "seed-1-step-1000",
            promptTokens: prompt.length,
            generatedTokens: length,
            options,
            runs,
            normal: { samples: normal, meanSeconds: normalSeconds, tokensPerSecond: normalRate },
            cached: { samples: cached, meanSeconds: cachedSeconds, tokensPerSecond: cachedRate },
            speedup: cachedRate / normalRate,
            identicalOutput: cachedOutput.join(",") === expected.join(","),
            sample: tokenizer.decode(cachedOutput),
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/inference-kv-cache.json");
