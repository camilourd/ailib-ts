// Teoría: docs/teoria/secuencias/multi-head.md
// Gráfico: proyecciones, dos cabezas con sus pesos, concatenación y máscara causal.

import { writeFileSync } from "node:fs";

import { Tensor } from "../../src/core/tensor.ts";
import { MultiHeadAttention } from "../../src/nn/multihead.ts";

const attention = new MultiHeadAttention(4, 2, { seed: 1 });
const input = Tensor.from([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0], [3, 4]);
const { weights } = attention.forward(input);

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">`,
    `<rect width="1000" height="560" fill="#ffffff"/>`,
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>`,
    `<text x="30" y="34" font-family="sans-serif" font-size="18" fill="#111827">Multi-head: dos cabezas, dos patrones de atención</text>`,
];

function box(x: number, y: number, width: number, height: number, fill: string, label: string): void {
    parts.push(
        `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="8" fill="${fill}" stroke="#94a3b8"/>`,
        `<text x="${x + width / 2}" y="${y + height / 2 + 4}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">${label}</text>`,
    );
}

function arrow(x1: number, y1: number, x2: number, y2: number): void {
    parts.push(
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#475569" stroke-width="1.6" marker-end="url(#arrow)"/>`,
    );
}

box(30, 100, 130, 70, "#eef2ff", "X  [3, 4]");
box(220, 60, 100, 36, "#dbeafe", "Q = XW_Q");
box(220, 120, 100, 36, "#dbeafe", "K = XW_K");
box(220, 180, 100, 36, "#dbeafe", "V = XW_V");
box(360, 60, 90, 156, "#f8fafc", "partir en h cabezas");
arrow(160, 135, 218, 78);
arrow(160, 135, 218, 138);
arrow(160, 135, 218, 198);
arrow(320, 78, 358, 120);
arrow(320, 138, 358, 138);
arrow(320, 198, 358, 156);

function heatmap(x: number, y: number, title: string, head: Tensor): void {
    parts.push(
        `<text x="${x + 54}" y="${y - 12}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">${title}</text>`,
    );
    const cell = 36;
    for (let row = 0; row < 3; row++) {
        for (let column = 0; column < 3; column++) {
            const value = head.get(row, column);
            const fill =
                value < 1e-6
                    ? "#f1f5f9"
                    : `rgba(31, 95, 168, ${(0.08 + 0.8 * value).toFixed(3)})`;
            parts.push(
                `<rect x="${x + column * cell}" y="${y + row * cell}" width="${cell}" height="${cell}" fill="${fill}" stroke="#ffffff" stroke-width="2"/>`,
            );
            if (value > 0.005) {
                parts.push(
                    `<text x="${x + column * cell + cell / 2}" y="${y + row * cell + cell / 2 + 4}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="${value > 0.55 ? "#ffffff" : "#1f2937"}">${value.toFixed(2)}</text>`,
                );
            }
        }
    }
    for (let index = 0; index < 3; index++) {
        parts.push(
            `<text x="${x - 8}" y="${y + index * cell + cell / 2 + 4}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#64748b">q${index}</text>`,
            `<text x="${x + index * cell + cell / 2}" y="${y - 26}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#64748b">k${index}</text>`,
        );
    }
}

heatmap(500, 110, "cabeza 0", weights[0] as Tensor);
heatmap(720, 110, "cabeza 1", weights[1] as Tensor);
arrow(450, 138, 480, 138);
parts.push(
    `<line x1="480" y1="78" x2="480" y2="198" stroke="#475569" stroke-width="1.6"/>`,
);
arrow(480, 88, 720, 88);
arrow(480, 108, 496, 108);

box(500, 300, 320, 50, "#dcfce7", "Concat(head_0, head_1)  [3, 4]");
box(860, 300, 110, 50, "#ffedd5", "W_O → salida");
arrow(554, 218, 554, 298);
arrow(774, 218, 774, 298);
arrow(820, 325, 858, 325);

parts.push(
    `<text x="30" y="250" font-family="sans-serif" font-size="13" fill="#111827">Cada cabeza: su propio Q/K/V y sus pesos.</text>`,
    `<text x="30" y="272" font-family="sans-serif" font-size="13" fill="#111827">La máscara causal apaga el futuro.</text>`,
    `<text x="30" y="380" font-family="sans-serif" font-size="14" fill="#111827">O = Concat(head_1, …, head_h) · W_O</text>`,
);

// Máscara causal 4×4: futuro apagado.
const maskX = 30;
const maskY = 430;
const maskCell = 26;
parts.push(
    `<text x="${maskX}" y="${maskY - 12}" font-family="sans-serif" font-size="12" fill="#111827">máscara causal (t = fila, k = columna)</text>`,
);
for (let row = 0; row < 4; row++) {
    for (let column = 0; column < 4; column++) {
        const allowed = column <= row;
        parts.push(
            `<rect x="${maskX + column * maskCell}" y="${maskY + row * maskCell}" width="${maskCell}" height="${maskCell}" fill="${allowed ? "#bfdbfe" : "#e2e8f0"}" stroke="#ffffff"/>`,
        );
    }
}
parts.push(
    `<text x="${maskX + 4 * maskCell + 16}" y="${maskY + 20}" font-family="sans-serif" font-size="11" fill="#475569">azul: puede mirar</text>`,
    `<text x="${maskX + 4 * maskCell + 16}" y="${maskY + 38}" font-family="sans-serif" font-size="11" fill="#475569">gris: futuro apagado</text>`,
    `</svg>`,
);

writeFileSync("docs/teoria/secuencias/multi-head.svg", `${parts.join("\n")}\n`, "utf8");
console.log("gráfico: docs/teoria/secuencias/multi-head.svg");
