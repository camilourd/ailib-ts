// Teoría: docs/teoria/entrenamiento/preentreno.md
// Gráfico: curvas de pérdida de train y validación del preentreno.

import { readFileSync, writeFileSync } from "node:fs";

interface Point {
    step: number;
    trainLoss: number;
    validationLoss: number;
    perplexity: number;
}

const summary = JSON.parse(readFileSync("data/pretrain-seed-1.json", "utf8")) as {
    seed: number;
    steps: number;
    history: Point[];
};
const history = summary.history;
const width = 900;
const height = 460;
const left = 70;
const right = 40;
const top = 50;
const bottom = 70;
const plotWidth = width - left - right;
const plotHeight = height - top - bottom;
const maxStep = Math.max(...history.map((point) => point.step), 1);
const losses = history.flatMap((point) => [point.trainLoss, point.validationLoss]);
const minLoss = Math.min(...losses);
const maxLoss = Math.max(...losses);
const padding = (maxLoss - minLoss) * 0.1 || 0.1;
const low = minLoss - padding;
const high = maxLoss + padding;

const x = (step: number): number => left + (step / maxStep) * plotWidth;
const y = (loss: number): number => top + (1 - (loss - low) / (high - low)) * plotHeight;

function path(key: "trainLoss" | "validationLoss"): string {
    return history
        .map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.step).toFixed(1)} ${y(point[key]).toFixed(1)}`)
        .join(" ");
}

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    `<rect width="${width}" height="${height}" fill="#ffffff"/>`,
    `<text x="${left}" y="30" font-family="sans-serif" font-size="17" fill="#111827">Preentreno: pérdida de train y validación</text>`,
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
    `<text x="20" y="${top + plotHeight / 2}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#475569" transform="rotate(-90 20 ${top + plotHeight / 2})">pérdida</text>`,
    `<path d="${path("trainLoss")}" fill="none" stroke="#2563eb" stroke-width="2"/>`,
    `<path d="${path("validationLoss")}" fill="none" stroke="#dc2626" stroke-width="2"/>`,
    `<line x1="${left + plotWidth - 170}" y1="24" x2="${left + plotWidth - 145}" y2="24" stroke="#2563eb" stroke-width="2"/>`,
    `<text x="${left + plotWidth - 138}" y="28" font-family="sans-serif" font-size="11" fill="#475569">train</text>`,
    `<line x1="${left + plotWidth - 80}" y1="24" x2="${left + plotWidth - 55}" y2="24" stroke="#dc2626" stroke-width="2"/>`,
    `<text x="${left + plotWidth - 48}" y="28" font-family="sans-serif" font-size="11" fill="#475569">validación</text>`,
    `</svg>`,
);

writeFileSync("docs/teoria/entrenamiento/preentreno.svg", `${parts.join("\n")}\n`, "utf8");
console.log(`curvas: ${history.length} puntos · ${minLoss.toFixed(3)} a ${maxLoss.toFixed(3)}`);
