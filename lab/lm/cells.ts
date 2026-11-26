// Teoría: docs/teoria/secuencias/gru-lstm.md
// Experimento: RNN, GRU y LSTM en el mismo LM; pérdida y gradiente por posición.

import { writeFileSync } from "node:fs";

import { Tensor } from "../../src/core/tensor.ts";
import { ByteTokenizer } from "../../src/data/tokenizer.ts";
import { CharLM } from "../../src/models/char-lm.ts";
import { crossEntropy } from "../../src/nn/losses.ts";
import type { CellType } from "../../src/nn/recurrent.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { TEXTO } from "./texto.ts";

const tokenizer = new ByteTokenizer();
const tokens = tokenizer.encode(TEXTO);
const steps = 3000;
const window = 33;
const length = 96;
const cells: CellType[] = ["rnn", "gru", "lstm"];

const norm = (tensor: Tensor): number => {
    const gradient = tensor.grad;
    if (!gradient) return 0;
    let total = 0;
    for (const value of gradient.data) total += value * value;
    return Math.sqrt(total);
};

const results = cells.map((cell) => {
    const model = new CharLM(256, { embeddingSize: 32, hiddenSize: 64, seed: 1, cell });
    const optimizer = new SGD(model.parameters(), 0.05);
    let lastLoss = 0;
    for (let step = 0; step < steps; step++) {
        const start = (step * 32) % (tokens.length - window);
        const piece = tokens.slice(start, start + window);
        optimizer.zeroGrad();
        const loss = model.loss(piece);
        loss.backward();
        optimizer.step();
        lastLoss = loss.get();
    }

    optimizer.zeroGrad();
    const piece = tokens.slice(0, length + 1);
    const inputs = piece.slice(0, -1);
    const { logits, states } = model.forward(inputs);
    const finalLoss = crossEntropy(logits.at(-1) as Tensor, [piece.at(-1) as number]);
    finalLoss.backward();
    const norms = states.map(norm);

    const start = tokenizer.encode("En un ");
    const generated = model.generate(start, 80, { temperature: 0.8, seed: 7 });
    return {
        cell,
        loss: lastLoss,
        norms,
        text: tokenizer.decode([...start, ...generated]),
    };
});

console.log("celda  pérdida  pos0      pos95     ratio");
for (const result of results) {
    const first = result.norms[0] as number;
    const last = result.norms[length - 1] as number;
    console.log(
        `${result.cell.padEnd(6)} ${result.loss.toFixed(4)}   ` +
            `${first.toExponential(2)}  ${last.toExponential(2)}  ` +
            `${(first / Math.max(last, 1e-12)).toExponential(2)}`,
    );
}
for (const result of results) {
    console.log(`\n${result.cell}: ${result.text}`);
}

const colors: Record<string, string> = {
    rnn: "#1f5fa8",
    gru: "#2f8f46",
    lstm: "#c26a1b",
};
const width = 880;
const height = 420;
const left = 70;
const right = 30;
const top = 50;
const bottom = 60;
const logs = results.flatMap((result) => result.norms.map((value) => Math.log10(Math.max(value, 1e-12))));
const minimum = Math.floor(Math.min(...logs));
const maximum = Math.ceil(Math.max(...logs));
const x = (position: number): number => left + (position / (length - 1)) * (width - left - right);
const y = (value: number): number =>
    top + ((maximum - value) / Math.max(maximum - minimum, 1)) * (height - top - bottom);

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
    `<text x="${left}" y="28" font-family="sans-serif" font-size="16" fill="#222">Gradiente por posición: RNN vs GRU vs LSTM</text>`,
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
for (const result of results) {
    const points = result.norms
        .map((value, position) => `${x(position)},${y(Math.log10(Math.max(value, 1e-12)))}`)
        .join(" ");
    parts.push(
        `<polyline points="${points}" fill="none" stroke="${colors[result.cell]}" stroke-width="2.5"/>`,
    );
}
results.forEach((result, index) => {
    const legendX = width - 190;
    const legendY = top + 18 + index * 18;
    parts.push(
        `<line x1="${legendX}" y1="${legendY}" x2="${legendX + 18}" y2="${legendY}" stroke="${colors[result.cell]}" stroke-width="3"/>`,
        `<text x="${legendX + 24}" y="${legendY + 4}" font-family="sans-serif" font-size="12" fill="#333">${result.cell.toUpperCase()}</text>`,
    );
});
parts.push(
    `<text x="${left}" y="${height - 12}" font-family="sans-serif" font-size="11" fill="#555">posición en la secuencia</text>`,
    `</svg>`,
);
const svgPath = "docs/teoria/secuencias/gru-lstm.svg";
writeFileSync(svgPath, `${parts.join("\n")}\n`, "utf8");
console.log(`\ncurva: ${svgPath}`);
