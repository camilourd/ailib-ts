// Teoría: docs/teoria/nucleo/bpe.md
// Entrena el BPE del corpus de F6 y guarda los merges.

import { readFileSync } from "node:fs";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { saveBPE } from "../../src/data/bpe-io.ts";

const text = readFileSync("data/quijote.txt", "utf8");
const split = Math.floor(text.length * 0.9);
const trainText = text.slice(0, split);
const validationText = text.slice(split);

const bpe = new BPETokenizer();
const start = performance.now();
bpe.train(trainText, { vocabularySize: 512, minFrequency: 2 });
const seconds = (performance.now() - start) / 1000;
saveBPE("data/quijote-bpe.json", bpe);

const trainTokens = bpe.encode(trainText);
const validationTokens = bpe.encode(validationText);
const trainCharacters = [...trainText].length;
const validationCharacters = [...validationText].length;

console.log(`BPE · vocabulario ${bpe.vocabularySize} · ${bpe.merges.length} merges · ${seconds.toFixed(1)} s`);
console.log(
    `train ${trainCharacters} caracteres → ${trainTokens.length} tokens ` +
        `(${(trainTokens.length / trainCharacters).toFixed(3)} tokens/carácter)`,
);
console.log(
    `val   ${validationCharacters} caracteres → ${validationTokens.length} tokens ` +
        `(${(validationTokens.length / validationCharacters).toFixed(3)} tokens/carácter)`,
);
console.log(`guardado: data/quijote-bpe.json`);
