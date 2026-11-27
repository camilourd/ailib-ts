// Teoría: docs/teoria/entrenamiento/sampling.md
// Experimento: temperatura, top-k y top-p sobre el mismo prompt.

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
const settings = [
    { name: "greedy (top-k 1)", options: { temperature: 0.8, topK: 1, seed: 7 } },
    { name: "temperatura 0.3", options: { temperature: 0.3, seed: 7 } },
    { name: "temperatura 0.8", options: { temperature: 0.8, seed: 7 } },
    { name: "temperatura 1.2", options: { temperature: 1.2, seed: 7 } },
    { name: "top-k 20", options: { temperature: 0.8, topK: 20, seed: 7 } },
    { name: "top-p 0.9", options: { temperature: 0.8, topP: 0.9, seed: 7 } },
];

const rows: Array<{ name: string; text: string }> = [];
for (const setting of settings) {
    const generated = model.generate(prompt, 48, { ...setting.options, useCache: true });
    const text = tokenizer.decode(generated);
    rows.push({ name: setting.name, text });
    console.log(`${setting.name}: ${text}`);
}

writeFileSync(
    "data/sampling-report.json",
    `${JSON.stringify({ model: "transformer-1.4m", promptTokens: prompt.length, rows }, null, 2)}\n`,
    "utf8",
);
console.log("guardado: data/sampling-report.json");
