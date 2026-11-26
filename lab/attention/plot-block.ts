// Teoría: docs/teoria/secuencias/bloque-transformer.md
// Gráfico: bloque pre-norm con dos sub-bloques y sus residuales.

import { writeFileSync } from "node:fs";

const parts: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560" viewBox="0 0 1000 560">`,
    `<rect width="1000" height="560" fill="#ffffff"/>`,
    `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>`,
    `<text x="30" y="34" font-family="sans-serif" font-size="18" fill="#111827">Bloque transformer pre-norm: dos sub-bloques con residual</text>`,
];

function box(
    x: number,
    y: number,
    width: number,
    height: number,
    fill: string,
    title: string,
    subtitle = "",
): void {
    parts.push(
        `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="9" fill="${fill}" stroke="#94a3b8"/>`,
        `<text x="${x + width / 2}" y="${y + (subtitle ? 20 : height / 2 + 4)}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">${title}</text>`,
    );
    if (subtitle) {
        parts.push(
            `<text x="${x + width / 2}" y="${y + 36}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#475569">${subtitle}</text>`,
        );
    }
}

function arrow(x1: number, y1: number, x2: number, y2: number, dashed = false): void {
    parts.push(
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#475569" stroke-width="1.6" ${dashed ? `stroke-dasharray="5 4"` : ""} marker-end="url(#arrow)"/>`,
    );
}

function add(x: number, y: number): void {
    parts.push(
        `<circle cx="${x}" cy="${y}" r="22" fill="#f3e8ff" stroke="#7e22ce" stroke-width="1.6"/>`,
        `<text x="${x}" y="${y + 6}" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#581c87">+</text>`,
    );
}

// Fila superior: x + MHA(LN(x)) = x1
box(30, 90, 110, 50, "#eef2ff", "x", "[secuencia, d]");
box(200, 90, 110, 50, "#fef9c3", "LayerNorm", "gamma, beta");
box(370, 90, 150, 50, "#dbeafe", "Multi-head", "causal, h cabezas");
add(590, 115);
box(660, 90, 110, 50, "#f5f3ff", "x1", "[secuencia, d]");
arrow(140, 115, 198, 115);
arrow(310, 115, 368, 115);
arrow(520, 115, 566, 115);
arrow(612, 115, 658, 115);
arrow(85, 90, 85, 55);
parts.push(
    `<line x1="85" y1="55" x2="590" y2="55" stroke="#475569" stroke-width="1.6"/>`,
);
arrow(590, 55, 590, 91);
parts.push(
    `<text x="300" y="46" font-family="sans-serif" font-size="11" fill="#475569">residual</text>`,
);

// x1 baja a la fila inferior
parts.push(
    `<line x1="770" y1="115" x2="770" y2="245" stroke="#475569" stroke-width="1.6" stroke-dasharray="5 4"/>`,
    `<line x1="770" y1="245" x2="85" y2="245" stroke="#475569" stroke-width="1.6" stroke-dasharray="5 4"/>`,
);
arrow(85, 245, 85, 298, true);

// Fila inferior: x1 + FFN(LN(x1)) = y
box(30, 300, 110, 50, "#f5f3ff", "x1", "[secuencia, d]");
box(200, 300, 110, 50, "#fef9c3", "LayerNorm", "gamma, beta");
box(370, 300, 150, 50, "#dcfce7", "FFN", "d → 4d → d");
add(590, 325);
box(660, 300, 110, 50, "#ffedd5", "y", "[secuencia, d]");
arrow(140, 325, 198, 325);
arrow(310, 325, 368, 325);
arrow(520, 325, 566, 325);
arrow(612, 325, 658, 325);
arrow(85, 350, 85, 405);
parts.push(
    `<line x1="85" y1="405" x2="590" y2="405" stroke="#475569" stroke-width="1.6"/>`,
);
arrow(590, 405, 590, 349);
parts.push(
    `<text x="300" y="425" font-family="sans-serif" font-size="11" fill="#475569">residual</text>`,
);

// Fórmulas
parts.push(
    `<text x="30" y="480" font-family="sans-serif" font-size="14" fill="#111827">x1 = x + MultiHead(LayerNorm(x))</text>`,
    `<text x="30" y="506" font-family="sans-serif" font-size="14" fill="#111827">y  = x1 + FFN(LayerNorm(x1))</text>`,
    `<text x="30" y="538" font-family="sans-serif" font-size="11" fill="#64748b">La línea punteada lleva x1 al segundo sub-bloque; las líneas largas son los residuales.</text>`,
    `</svg>`,
);

writeFileSync("docs/teoria/secuencias/bloque-transformer.svg", `${parts.join("\n")}\n`, "utf8");
console.log("gráfico: docs/teoria/secuencias/bloque-transformer.svg");
