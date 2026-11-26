// Teoría: docs/teoria/secuencias/rope.md
// Gráfico: rotación de un par de dimensiones según la posición.

import { writeFileSync } from "node:fs";

const centerX = 230;
const centerY = 250;
const radius = 130;
const angle = 1;

const arcPoints: string[] = [];
for (let step = 0; step <= 20; step++) {
    const current = (angle * step) / 20;
    arcPoints.push(
        `${(centerX + radius * Math.cos(current)).toFixed(1)},${(centerY - radius * Math.sin(current)).toFixed(1)}`,
    );
}

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="920" height="500" viewBox="0 0 920 500">`,
    `<rect width="920" height="500" fill="#ffffff"/>`,
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>`,
    `<text x="30" y="34" font-family="sans-serif" font-size="18" fill="#111827">RoPE: girar cada par de dimensiones según la posición</text>`,
    `<line x1="${centerX - radius - 30}" y1="${centerY}" x2="${centerX + radius + 30}" y2="${centerY}" stroke="#cbd5e1"/>`,
    `<line x1="${centerX}" y1="${centerY - radius - 30}" x2="${centerX}" y2="${centerY + radius + 30}" stroke="#cbd5e1"/>`,
    `<circle cx="${centerX}" cy="${centerY}" r="${radius}" fill="none" stroke="#e2e8f0"/>`,
    `<polyline points="${arcPoints.join(" ")}" fill="none" stroke="#f59e0b" stroke-width="2"/>`,
    `<line x1="${centerX}" y1="${centerY}" x2="${centerX + radius}" y2="${centerY}" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>`,
    `<line x1="${centerX}" y1="${centerY}" x2="${(centerX + radius * Math.cos(angle)).toFixed(1)}" y2="${(centerY - radius * Math.sin(angle)).toFixed(1)}" stroke="#1f5fa8" stroke-width="3" marker-end="url(#arrow)"/>`,
    `<text x="${centerX + radius + 8}" y="${centerY + 5}" font-family="sans-serif" font-size="13" fill="#475569">x</text>`,
    `<text x="${(centerX + radius * Math.cos(angle) + 8).toFixed(1)}" y="${(centerY - radius * Math.sin(angle) - 8).toFixed(1)}" font-family="sans-serif" font-size="13" fill="#1f5fa8">R(θ)·x</text>`,
    `<text x="${centerX + 82}" y="${centerY - 42}" font-family="sans-serif" font-size="13" fill="#b45309">θ</text>`,
    `<text x="450" y="120" font-family="sans-serif" font-size="14" fill="#111827">θ_i = base^(−2i/d)</text>`,
    `<text x="450" y="156" font-family="sans-serif" font-size="13" fill="#111827">x'_{2i}   = x_{2i}·cos(p·θ_i) − x_{2i+1}·sin(p·θ_i)</text>`,
    `<text x="450" y="182" font-family="sans-serif" font-size="13" fill="#111827">x'_{2i+1} = x_{2i}·sin(p·θ_i) + x_{2i+1}·cos(p·θ_i)</text>`,
    `<text x="450" y="230" font-family="sans-serif" font-size="13" fill="#111827">La norma del par no cambia: es una rotación.</text>`,
    `<text x="450" y="256" font-family="sans-serif" font-size="13" fill="#111827">q_i · k_j depende de la distancia i − j, no del origen.</text>`,
    `<text x="450" y="300" font-family="sans-serif" font-size="12" fill="#475569">Posición 0: sin giro. Posición p: gira p·θ_i.</text>`,
    `<text x="450" y="324" font-family="sans-serif" font-size="12" fill="#475569">Se aplica a Q y K antes de la atención; V no se rota.</text>`,
    `<text x="450" y="348" font-family="sans-serif" font-size="12" fill="#475569">Así la atención puede distinguir orden sin sumar posiciones absolutas.</text>`,
    `<text x="30" y="450" font-family="sans-serif" font-size="13" fill="#111827">En el laboratorio: posición 1 → [0.540, 0.841]; posición 2 → [−0.416, 0.909].</text>`,
    `<text x="30" y="476" font-family="sans-serif" font-size="13" fill="#111827">q·k a distancia 2 da 0.79371 desde dos orígenes distintos.</text>`,
    `</svg>`,
];

writeFileSync("docs/teoria/secuencias/rope.svg", `${parts.join("\n")}\n`, "utf8");
console.log("gráfico: docs/teoria/secuencias/rope.svg");
