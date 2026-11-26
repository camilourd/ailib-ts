// Prepara el extracto de Don Quijote desde el texto completo descargado de Gutenberg.
// El archivo original está en ISO-8859-1; se convierte a UTF-8.

import { readFileSync, writeFileSync } from "node:fs";

const raw = readFileSync("data/quijote-full.txt");
let text = raw.toString("latin1");

const startMarker = "*** START OF THIS PROJECT GUTENBERG EBOOK DON QUIJOTE ***";
const endMarker = "*** END OF THIS PROJECT GUTENBERG EBOOK DON QUIJOTE ***";
const start = text.indexOf(startMarker);
if (start >= 0) text = text.slice(start + startMarker.length);
const end = text.indexOf(endMarker);
if (end >= 0) text = text.slice(0, end);

const novel = text.indexOf("En un lugar de la Mancha");
if (novel >= 0) text = text.slice(novel);

const excerpt = text.slice(0, 200000).trim() + "\n";
writeFileSync("data/quijote.txt", excerpt, "utf8");

const characters = [...excerpt].length;
console.log(`data/quijote.txt · ${characters} caracteres · ${excerpt.length} bytes UTF-8`);
console.log(excerpt.split("\n").slice(0, 6).join("\n"));
