// F9.1: un genoma mínimo produce un tokenizer lossless sobre todo el corpus.
// Se comparan palabra y carácter lossless; el diccionario/BPE llegan en la evolución.

import { readFileSync, writeFileSync } from "node:fs";

import {
    buildTokenizer,
    isLossless,
    programFromGenome,
} from "../../src/data/tokenizer-program.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const trainEnd = Math.floor(quijote.length * 0.7);
const validationEnd = Math.floor(quijote.length * 0.85);
const train = quijote.slice(0, trainEnd);
const validation = quijote.slice(trainEnd, validationEnd);
const test = quijote.slice(validationEnd);

const genomes = [
    { name: "palabra-512", genome: [2, 512, 32, 2] },
    { name: "carácter-512", genome: [1, 512, 32, 1] },
];

const rows = [];
for (const { name, genome } of genomes) {
    const program = programFromGenome(genome);
    const tokenizer = buildTokenizer(program, train);
    const exact = isLossless(tokenizer, [train, validation, test, external]);
    const ratio = (text: string): number => tokenizer.encode(text).length / [...text].length;
    rows.push({
        name,
        genome,
        program,
        vocabulary: tokenizer.vocabularySize,
        exact,
        train: ratio(train),
        validation: ratio(validation),
        test: ratio(test),
        external: ratio(external),
    });
    console.log(
        `${name}: vocab ${tokenizer.vocabularySize} · exacto ${exact ? "sí" : "no"} · ` +
            `val ${ratio(validation).toFixed(3)} · externo ${ratio(external).toFixed(3)}`,
    );
}

writeFileSync(
    "data/tokenizer-program.json",
    `${JSON.stringify({ genomes: rows }, null, 2)}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-program.json");
