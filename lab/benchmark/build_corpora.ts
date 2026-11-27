// F12.0: construye tres corpus con split 70/15/15 por caracteres.
// - narrativo: Quijote
// - técnico: documentación del repo
// - código: src + test TypeScript

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join, extname } from "node:path";

const OUT = "data/corpora";

function collect(directory: string, extensions: Set<string>): string[] {
    const files: string[] = [];
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) {
            files.push(...collect(path, extensions));
        } else if (extensions.has(extname(entry.name))) {
            files.push(path);
        }
    }
    return files.sort();
}

function split(text: string): { train: string; validation: string; test: string } {
    const first = Math.floor(text.length * 0.7);
    const second = Math.floor(text.length * 0.85);
    return {
        train: text.slice(0, first),
        validation: text.slice(first, second),
        test: text.slice(second),
    };
}

function writeCorpus(name: string, text: string): void {
    mkdirSync(OUT, { recursive: true });
    const parts = split(text);
    for (const [part, value] of Object.entries(parts)) {
        writeFileSync(`${OUT}/${name}-${part}.txt`, value, "utf8");
    }
    console.log(`${name}: ${text.length} caracteres · train ${parts.train.length} · val ${parts.validation.length} · test ${parts.test.length}`);
}

const quijote = readFileSync("data/quijote.txt", "utf8");
const technicalFiles = [
    ...collect("docs/teoria", new Set([".md"])),
    ...collect("docs/experimentos", new Set([".md"])),
];
const codeFiles = [
    ...collect("src", new Set([".ts"])),
    ...collect("test", new Set([".ts"])),
];
const technical = technicalFiles.map((file) => readFileSync(file, "utf8")).join("\n\n");
const code = codeFiles.map((file) => readFileSync(file, "utf8")).join("\n\n");

writeCorpus("narrativo", quijote);
writeCorpus("tecnico", technical);
writeCorpus("codigo", code);

writeFileSync(
    `${OUT}/README.md`,
    `# Corpus F12\n\n- narrativo: data/quijote.txt\n- técnico: ${technicalFiles.length} archivos .md de docs/\n- código: ${codeFiles.length} archivos .ts de src/ y test/\n\nSplit por caracteres: 70% train, 15% validación, 15% test. El tokenizer se entrena solo con train.\n`,
    "utf8",
);
