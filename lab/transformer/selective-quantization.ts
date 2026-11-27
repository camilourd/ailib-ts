// Teoría: docs/teoria/entrenamiento/f8-cuantizacion-selectiva.md
// Pregunta propia: ¿qué módulos toleran INT8 sin perder perplejidad?

import { readFileSync, writeFileSync } from "node:fs";

import { loadBPE } from "../../src/data/bpe-io.ts";
import { loadCheckpoint } from "../../src/io/checkpoint.ts";
import { TransformerLM } from "../../src/models/transformer-lm.ts";
import {
    dequantizeInt8,
    floatBytes,
    quantizeInt8,
    quantizedBytes,
} from "../../src/nn/quantization.ts";

type Group = "embedding" | "attention" | "ffn" | "norms" | "output";

const GROUP_NAMES: Group[] = ["embedding", "attention", "ffn", "norms", "output"];

function groupOf(name: string): Group {
    if (name === "embedding.weights") return "embedding";
    if (name.startsWith("output.")) return "output";
    if (name.startsWith("finalNorm")) return "norms";
    const match = /^block\.\d+\.(\d+)$/.exec(name);
    if (!match) throw new Error(`parámetro sin grupo: ${name}`);
    const position = Number(match[1]);
    if (position >= 2 && position <= 9) return "attention";
    if (position >= 12) return "ffn";
    return "norms";
}

const tokenizer = loadBPE("data/quijote-bpe.json");
const text = readFileSync("data/quijote.txt", "utf8");
const split = Math.floor(text.length * 0.9);
const validationTokens = tokenizer.encode(text.slice(split));
const context = 128;
const windows = 8;

const model = new TransformerLM(512, {
    modelSize: 160,
    heads: 8,
    blocks: 4,
    hiddenSize: 640,
    seed: 1,
});
const checkpoint = loadCheckpoint("checkpoints/seed-1-step-1000.json", { requiresGrad: true });
model.loadParameters(checkpoint.parameters);

function evaluate(): number {
    const stride = Math.max(1, Math.floor((validationTokens.length - context - 1) / windows));
    let total = 0;
    for (let index = 0; index < windows; index++) {
        const start = index * stride;
        total += model.loss(validationTokens.slice(start, start + context + 1)).get();
    }
    return Math.exp(total / windows);
}

const named = model.namedParameters();
const backup = new Map(
    Object.entries(named).map(([name, tensor]) => [name, new Float32Array(tensor.data)]),
);
const floatTotal = Object.values(named).reduce((total, tensor) => total + floatBytes(tensor), 0);

function restore(): void {
    for (const [name, tensor] of Object.entries(named)) {
        tensor.data.set(backup.get(name) as Float32Array);
    }
}

const all = new Set<Group>(GROUP_NAMES);
const variants: Array<{ name: string; groups: Set<Group> }> = [
    { name: "float32", groups: new Set() },
    { name: "todo INT8", groups: all },
    { name: "solo atención", groups: new Set(["attention"]) },
    { name: "solo FFN", groups: new Set(["ffn"]) },
    { name: "solo embedding+salida", groups: new Set(["embedding", "output"]) },
    { name: "todo menos atención", groups: new Set(["embedding", "ffn", "norms", "output"]) },
    { name: "todo menos FFN", groups: new Set(["embedding", "attention", "norms", "output"]) },
    { name: "todo menos embedding/salida", groups: new Set(["attention", "ffn", "norms"]) },
    { name: "todo menos normas", groups: new Set(["embedding", "attention", "ffn", "output"]) },
];

const rows: Array<{
    name: string;
    bytes: number;
    ratio: number;
    perplexity: number;
}> = [];
for (const variant of variants) {
    restore();
    let bytes = 0;
    for (const [name, tensor] of Object.entries(named)) {
        if (variant.groups.has(groupOf(name))) {
            const quantized = quantizeInt8(tensor, { perChannel: true });
            tensor.data.set(dequantizeInt8(quantized).data);
            bytes += quantizedBytes(quantized);
        } else {
            bytes += floatBytes(tensor);
        }
    }
    rows.push({
        name: variant.name,
        bytes,
        ratio: floatTotal / bytes,
        perplexity: evaluate(),
    });
}
restore();

console.log("pregunta: ¿qué módulos toleran INT8 sin perder perplejidad?");
console.log("variante                    MB     ratio   perplejidad");
for (const row of rows) {
    console.log(
        `${row.name.padEnd(27)} ${(row.bytes / 1024 / 1024).toFixed(2)}   ` +
            `${row.ratio.toFixed(2)}×   ${row.perplexity.toFixed(2)}`,
    );
}

writeFileSync(
    "data/selective-quantization.json",
    `${JSON.stringify(
        {
            question: "¿Qué módulos del transformer toleran INT8 sin perder perplejidad?",
            hypothesis:
                "Atención y FFN toleran INT8; embedding/salida/normas concentran la degradación.",
            model: model.architecture,
            context,
            windows,
            floatBytes: floatTotal,
            variants: rows,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/selective-quantization.json");
