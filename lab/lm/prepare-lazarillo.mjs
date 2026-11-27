// Prepara el texto externo de Lazarillo de Tormes (Gutenberg #320).

import { readFileSync, writeFileSync } from "node:fs";

const raw = readFileSync("data/lazarillo-full.txt");
let text = raw.toString("latin1");
const startMarker = "*** START OF THIS PROJECT GUTENBERG EBOOK";
const endMarker = "*** END OF THIS PROJECT GUTENBERG EBOOK";
const start = text.indexOf(startMarker);
if (start >= 0) {
    const lineEnd = text.indexOf("\n", start);
    text = text.slice(lineEnd + 1);
}
const novel = text.indexOf("LA VIDA DE LAZARILLO");
if (novel >= 0) text = text.slice(novel);
const end = text.indexOf(endMarker);
if (end >= 0) text = text.slice(0, end);
const excerpt = text.trim().slice(0, 100000) + "\n";
writeFileSync("data/lazarillo.txt", excerpt, "utf8");
console.log(`data/lazarillo.txt · ${[...excerpt].length} caracteres · ${excerpt.length} bytes`);
