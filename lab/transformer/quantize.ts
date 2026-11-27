// Teoría: docs/teoria/entrenamiento/cuantizacion.md
// Experimento: pesos INT8 por canal y por tensor frente a float32.

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

const baseline = evaluate();
const named = model.namedParameters();
const backup = new Map(
    Object.entries(named).map(([name, tensor]) => [name, new Float32Array(tensor.data)]),
);

function quantizeAll(perChannel: boolean): { floatTotal: number; quantTotal: number } {
    let floatTotal = 0;
    let quantTotal = 0;
    for (const [name, tensor] of Object.entries(named)) {
        const quantized = quantizeInt8(tensor, { perChannel });
        const restored = dequantizeInt8(quantized);
        tensor.data.set(restored.data);
        floatTotal += floatBytes(tensor);
        quantTotal += quantizedBytes(quantized);
        if (name.length === 0) throw new Error("parámetro sin nombre");
    }
    return { floatTotal, quantTotal };
}

function restore(): void {
    for (const [name, tensor] of Object.entries(named)) {
        tensor.data.set(backup.get(name) as Float32Array);
    }
}

const perChannelStorage = quantizeAll(true);
const perChannel = evaluate();
restore();
const perTensorStorage = quantizeAll(false);
const perTensor = evaluate();
restore();

const ratio = perChannelStorage.floatTotal / perChannelStorage.quantTotal;
console.log(
    `float32: ${(perChannelStorage.floatTotal / 1024 / 1024).toFixed(2)} MB · ppl ${baseline.toFixed(2)}`,
);
console.log(
    `INT8 por canal: ${(perChannelStorage.quantTotal / 1024 / 1024).toFixed(2)} MB · ` +
        `ratio ${ratio.toFixed(2)}× · ppl ${perChannel.toFixed(2)}`,
);
console.log(
    `INT8 por tensor: ${(perTensorStorage.quantTotal / 1024 / 1024).toFixed(2)} MB · ` +
        `ratio ${(perTensorStorage.floatTotal / perTensorStorage.quantTotal).toFixed(2)}× · ` +
        `ppl ${perTensor.toFixed(2)}`,
);

writeFileSync(
    "data/quantization-report.json",
    `${JSON.stringify(
        {
            model: "transformer-1.4m",
            parameters: model.parameterCount(),
            context,
            windows,
            float32: { bytes: perChannelStorage.floatTotal, perplexity: baseline },
            int8PerChannel: {
                bytes: perChannelStorage.quantTotal,
                ratio,
                perplexity: perChannel,
            },
            int8PerTensor: {
                bytes: perTensorStorage.quantTotal,
                ratio: perTensorStorage.floatTotal / perTensorStorage.quantTotal,
                perplexity: perTensor,
            },
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/quantization-report.json");
