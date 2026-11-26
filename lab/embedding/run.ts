// Teoría: docs/teoria/redes/embeddings.md
// Experimento: tokeniza un texto, busca sus filas y comprueba el gradiente disperso.

import { sum } from "../../src/core/tensor-ops.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { Embedding } from "../../src/nn/embedding.ts";

const tokenizer = new ByteTokenizer();
const text = "hola, ñandú 🐢";
const tokens = tokenizer.encode(text);
const embedding = new Embedding(256, 4, { seed: 1 });
const vectors = embedding.forward(tokens);
sum(vectors).backward();

const gradient = embedding.weights.grad;
let touched = 0;
for (const token of new Set(tokens)) {
    const row = [...(gradient?.data ?? [])].slice(token * 4, token * 4 + 4);
    if (row.some((value) => value !== 0)) touched += 1;
}

console.log(`texto: ${JSON.stringify(text)}`);
console.log(`tokens: ${tokens.length} · primeros: [${tokens.slice(0, 8).join(", ")}]`);
console.log(`ida y vuelta: ${JSON.stringify(tokenizer.decode(tokens))}`);
console.log(`vectores: ${JSON.stringify(vectors.shape)} · filas con gradiente: ${touched}`);
