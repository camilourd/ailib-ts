// Teoría: docs/teoria/secuencias/desvanecimiento.md
// Experimento: norma del gradiente de cada estado oculto a lo largo de una secuencia.

import { writeFileSync } from "node:fs";

import { Tensor } from "../../src/core/tensor.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharLM } from "../../src/models/char-lm.ts";
import { crossEntropy } from "../../src/nn/losses.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { TEXTO } from "./texto.ts";

const tokenizer = new ByteTokenizer();
const tokens = tokenizer.encode(TEXTO);
const model = new CharLM(256, { embeddingSize: 32, hiddenSize: 64, seed: 3 });
const optimizer = new SGD(model.parameters(), 0.05);

const window = 33;
for (let step = 0; step < 1500; step++) {
    const start = (step * 32) % (tokens.length - window);
    const piece = tokens.slice(start, start + window);
    optimizer.zeroGrad();
    const loss = model.loss(piece);
    loss.backward();
    optimizer.step();
}

optimizer.zeroGrad();
const length = 96;
const piece = tokens.slice(0, length + 1);
const inputs = piece.slice(0, -1);
const { logits, states } = model.forward(inputs);
const loss = crossEntropy(logits.at(-1) as Tensor, [piece.at(-1) as number]);
loss.backward();

const norm = (tensor: Tensor): number => {
    const gradient = tensor.grad;
    if (!gradient) return 0;
    let total = 0;
    for (const value of gradient.data) total += value * value;
    return Math.sqrt(total);
};
const norms = states.map(norm);
const selected = [0, 1, 7, 15, 31, 47, 63, 79, 95];
console.log(`pérdida de la ventana de ${length} pasos: ${loss.get().toFixed(4)}`);
for (const position of selected) {
    console.log(
        `posición ${String(position).padStart(2)}  norma ${(norms[position] as number).toFixed(6)}  ` +
            `log10 ${Math.log10(Math.max(norms[position] as number, 1e-12)).toFixed(2)}`,
    );
}
console.log(
    `cociente posición 0 / última: ` +
        `${((norms[0] as number) / Math.max(norms[length - 1] as number, 1e-12)).toExponential(2)}`,
);

const width = 880;
const height = 420;
const left = 70;
const right = 30;
const top = 50;
const bottom = 60;
const logs = norms.map((value) => Math.log10(Math.max(value, 1e-12)));
const minimum = Math.floor(Math.min(...logs));
const maximum = Math.ceil(Math.max(...logs));
const x = (position: number): number => left + (position / (length - 1)) * (width - left - right);
const y = (value: number): number =>
    top + ((maximum - value) / Math.max(maximum - minimum, 1)) * (height - top - bottom);

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
    `<text x="${left}" y="28" font-family="sans-serif" font-size="16" fill="#222">Norma del gradiente por posición (escala log10)</text>`,
];
for (let exponent = minimum; exponent <= maximum; exponent++) {
    parts.push(
        `<line x1="${left}" y1="${y(exponent)}" x2="${width - right}" y2="${y(exponent)}" stroke="#dddddd"/>`,
        `<text x="${left - 10}" y="${y(exponent) + 4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#555">1e${exponent}</text>`,
    );
}
for (const position of [0, 16, 32, 48, 64, 80, 95]) {
    parts.push(
        `<line x1="${x(position)}" y1="${top}" x2="${x(position)}" y2="${height - bottom}" stroke="#eeeeee"/>`,
        `<text x="${x(position)}" y="${height - bottom + 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">${position}</text>`,
    );
}
const points = logs.map((value, position) => `${x(position)},${y(value)}`).join(" ");
parts.push(
    `<polyline points="${points}" fill="none" stroke="#1f5fa8" stroke-width="2.5"/>`,
    `<text x="${left}" y="${height - 12}" font-family="sans-serif" font-size="11" fill="#555">posición en la secuencia</text>`,
    `</svg>`,
);
const svgPath = "docs/teoria/secuencias/desvanecimiento.svg";
writeFileSync(svgPath, `${parts.join("\n")}\n`, "utf8");
console.log(`curva: ${svgPath}`);
