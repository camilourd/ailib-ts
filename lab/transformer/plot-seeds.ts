// Teoría: docs/teoria/entrenamiento/f6-informe.md
// Gráfico: pérdida de validación de las dos semillas del preentreno.

import { readFileSync, writeFileSync } from "node:fs";

interface Point {
    step: number;
    validationLoss: number;
}

interface Summary {
    seed: number;
    steps: number;
    finalPerplexity: number;
    history: Point[];
}

const first = JSON.parse(readFileSync("data/pretrain-seed-1.json", "utf8")) as Summary;
const second = JSON.parse(readFileSync("data/pretrain-seed-2.json", "utf8")) as Summary;
const series = [first, second];
const width = 900;
const height = 460;
const left = 70;
const right = 40;
const top = 50;
const bottom = 70;
const plotWidth = width - left - right;
const plotHeight = height - top - bottom;
const maxStep = Math.max(...series.flatMap((item) => item.history.map((point) => point.step)), 1);
const losses = series.flatMap((item) => item.history.map((point) => point.validationLoss));
const minLoss = Math.min(...losses);
const maxLoss = Math.max(...losses);
const padding = (maxLoss - minLoss) * 0.1 || 0.1;
const low = minLoss - padding;
const high = maxLoss + padding;

const x = (step: number): number => left + (step / maxStep) * plotWidth;
const y = (loss: number): number => top + (1 - (loss - low) / (high - low)) * plotHeight;

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
    `<text x="${left}" y="30" font-family="sans-serif" font-size="17" fill="#111827">Preentreno: dos semillas, misma configuración</text>`,
    `<rect x="${left}" y="${top}" width="${plotWidth}" height="${plotHeight}" fill="#f8fafc" stroke="#cbd5e1"/>`,
];

for (let line = 0; line <= 5; line++) {
    const value = low + ((high - low) * line) / 5;
    const yLine = y(value);
    parts.push(
        `<line x1="${left}" y1="${yLine.toFixed(1)}" x2="${left + plotWidth}" y2="${yLine.toFixed(1)}" stroke="#e2e8f0"/>`,
        `<text x="${left - 10}" y="${(yLine + 4).toFixed(1)}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#64748b">${value.toFixed(2)}</text>`,
    );
}
for (let tick = 0; tick <= 4; tick++) {
    const step = Math.round((maxStep * tick) / 4);
    const xTick = x(step);
    parts.push(
        `<line x1="${xTick.toFixed(1)}" y1="${top}" x2="${xTick.toFixed(1)}" y2="${top + plotHeight}" stroke="#e2e8f0"/>`,
        `<text x="${xTick.toFixed(1)}" y="${top + plotHeight + 18}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#64748b">${step}</text>`,
    );
}
parts.push(
    `<text x="${left + plotWidth / 2}" y="${height - 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569">paso</text>`,
    `<text x="20" y="${top + plotHeight / 2}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569" transform="rotate(-90 20 ${top + plotHeight / 2})">pérdida de validación</text>`,
);
const colors = ["#2563eb", "#dc2626"];
series.forEach((summary, index) => {
    const path = summary.history
        .map((point, position) => `${position === 0 ? "M" : "L"} ${x(point.step).toFixed(1)} ${y(point.validationLoss).toFixed(1)}`)
        .join(" ");
    parts.push(`<path d="${path}" fill="none" stroke="${colors[index]}" stroke-width="2"/>`);
});
parts.push(
    `<line x1="${left + plotWidth - 190}" y1="24" x2="${left + plotWidth - 165}" y2="24" stroke="${colors[0]}" stroke-width="2"/>`,
    `<text x="${left + plotWidth - 158}" y="28" font-family="sans-serif" font-size="11" fill="#475569">semilla ${first.seed} · ppl ${first.finalPerplexity.toFixed(1)}</text>`,
    `<line x1="${left + plotWidth - 190}" y1="40" x2="${left + plotWidth - 165}" y2="40" stroke="${colors[1]}" stroke-width="2"/>`,
    `<text x="${left + plotWidth - 158}" y="44" font-family="sans-serif" font-size="11" fill="#475569">semilla ${second.seed} · ppl ${second.finalPerplexity.toFixed(1)}</text>`,
    `</svg>`,
);

writeFileSync("docs/teoria/entrenamiento/preentreno-semillas.svg", `${parts.join("\n")}\n`, "utf8");
console.log(
    `semillas: 1 ppl ${first.finalPerplexity.toFixed(2)} · 2 ppl ${second.finalPerplexity.toFixed(2)}`,
);
