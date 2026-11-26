// Teoría: docs/teoria/redes/sigmoid.md
// Experimento: genera docs/teoria/redes/sigmoid.svg con σ y σ′.

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { sigmoid, sigmoidDerivative } from "../../src/nn/sigmoid.ts";

const WIDTH = 760;
const HEIGHT = 340;
const MARGIN = 42;
const GAP = 64;
const PANEL_WIDTH = (WIDTH - 2 * MARGIN - GAP) / 2;
const PANEL_HEIGHT = HEIGHT - 2 * MARGIN;
const BOTTOM = MARGIN + PANEL_HEIGHT;
const X_MIN = -8;
const X_MAX = 8;
const SAMPLES = 401;

type Point = { x: number; y: number };

const scaleX = (x: number, left: number): number =>
    left + ((x - X_MIN) / (X_MAX - X_MIN)) * PANEL_WIDTH;

const scaleY = (y: number, maxY: number): number => BOTTOM - (y / maxY) * PANEL_HEIGHT;

const pathOf = (points: Point[], left: number, maxY: number): string =>
    points
        .map((point, index) => {
            const x = scaleX(point.x, left).toFixed(2);
            const y = scaleY(point.y, maxY).toFixed(2);
            return `${index === 0 ? "M" : "L"}${x},${y}`;
        })
        .join(" ");

function panel(options: {
    left: number;
    maxY: number;
    title: string;
    color: string;
    points: Point[];
    yTicks: number[];
}): string {
    const { left, maxY, title, color, points, yTicks } = options;
    const parts: string[] = [];
    parts.push(
        `<rect x="${left}" y="${MARGIN}" width="${PANEL_WIDTH}" height="${PANEL_HEIGHT}" fill="#ffffff" stroke="#e3e3e3"/>`,
    );
    for (const tick of yTicks) {
        const y = scaleY(tick, maxY);
        parts.push(`<line x1="${left}" y1="${y}" x2="${left + PANEL_WIDTH}" y2="${y}" stroke="#eeeeee"/>`);
        parts.push(`<text x="${left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#666">${tick}</text>`);
    }
    for (const x of [-8, -4, 0, 4, 8]) {
        const px = scaleX(x, left);
        parts.push(`<line x1="${px}" y1="${BOTTOM}" x2="${px}" y2="${BOTTOM + 4}" stroke="#999"/>`);
        parts.push(`<text x="${px}" y="${BOTTOM + 18}" text-anchor="middle" font-size="11" fill="#666">${x}</text>`);
    }
    parts.push(`<line x1="${left}" y1="${MARGIN}" x2="${left}" y2="${BOTTOM}" stroke="#333"/>`);
    parts.push(`<line x1="${left}" y1="${BOTTOM}" x2="${left + PANEL_WIDTH}" y2="${BOTTOM}" stroke="#333"/>`);
    parts.push(
        `<path d="${pathOf(points, left, maxY)}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round"/>`,
    );
    parts.push(
        `<text x="${left + PANEL_WIDTH / 2}" y="${MARGIN - 12}" text-anchor="middle" font-size="13" font-weight="600" fill="#222">${title}</text>`,
    );
    parts.push(
        `<text x="${left + PANEL_WIDTH / 2}" y="${BOTTOM + 34}" text-anchor="middle" font-size="11" fill="#666">x</text>`,
    );
    return parts.join("\n");
}

const marker = (
    panelLeft: number,
    maxY: number,
    value: number,
    label?: string,
    labelDy = -8,
): string => {
    const cx = scaleX(0, panelLeft);
    const cy = scaleY(value, maxY);
    const dot = `<circle cx="${cx}" cy="${cy}" r="4" fill="#ffffff" stroke="#111" stroke-width="2"/>`;
    const text = `<text x="${cx + 8}" y="${cy + labelDy}" font-size="11" fill="#111">${label}</text>`;
    return label ? [dot, text].join("\n") : dot;
};

const samples = Array.from(
    { length: SAMPLES },
    (_, index) => X_MIN + ((X_MAX - X_MIN) * index) / (SAMPLES - 1),
);
const sigmoidPoints = samples.map((x) => ({ x, y: sigmoid(x) }));
const derivativePoints = samples.map((x) => ({ x, y: sigmoidDerivative(x) }));

const left = MARGIN;
const right = MARGIN + PANEL_WIDTH + GAP;

const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-label="sigma y su derivada" font-family="system-ui, sans-serif">`,
    `<rect width="${WIDTH}" height="${HEIGHT}" fill="#fafafa"/>`,
    panel({ left, maxY: 1, title: "σ(x)", color: "#1f6feb", points: sigmoidPoints, yTicks: [0, 0.25, 0.5, 0.75, 1] }),
    marker(left, 1, 0.5, "0.5"),
    panel({ left: right, maxY: 0.25, title: "σ′(x)", color: "#d1242f", points: derivativePoints, yTicks: [0, 0.125, 0.25] }),
    marker(right, 0.25, 0.25),
    `<text x="${WIDTH / 2}" y="${HEIGHT - 8}" text-anchor="middle" font-size="11" fill="#888">σ tiende a 0 y 1; σ′ se apaga en las colas</text>`,
    `</svg>`,
].join("\n");

const target = new URL("../../docs/teoria/redes/sigmoid.svg", import.meta.url);
writeFileSync(target, `${svg}\n`, "utf8");
console.log(`gráfica escrita en ${fileURLToPath(target)}`);
