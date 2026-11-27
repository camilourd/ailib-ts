// F9.3a: tokenizer contextual con DP lossless; compara contexto y lookahead.
// Fitness barato: tokens/carácter en validación y Lazarillo.

import { readFileSync, writeFileSync } from "node:fs";

import { ContextTokenizer } from "../../src/data/context-tokenizer.ts";
import { LosslessByteTokenizer } from "../../src/data/lossless.ts";

const quijote = readFileSync("data/quijote.txt", "utf8");
const external = readFileSync("data/lazarillo.txt", "utf8");
const train = quijote.slice(0, 20000);
const validation = quijote.slice(20000, 28000);
const externalText = external.slice(0, 10000);
const UNIT = /\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+/gu;
const bytes = (value: string): number => new TextEncoder().encode(value).length;

const counts = new Map<string, number>();
const unitList = train.match(UNIT) ?? [];
for (const unit of unitList) {
    const characters = [...unit];
    for (let start = 0; start < characters.length; start++) {
        let candidate = "";
        for (
            let end = start;
            end < characters.length && bytes(candidate + characters[end]) <= 64;
            end++
        ) {
            candidate += characters[end] as string;
            counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
        }
    }
}
for (let index = 0; index < unitList.length; index++) {
    let candidate = "";
    for (
        let next = index;
        next < unitList.length && next < index + 4 && bytes(candidate + unitList[next]) <= 64;
        next++
    ) {
        candidate += unitList[next] as string;
        counts.set(candidate, (counts.get(candidate) ?? 0) + 1);
    }
}

const tokenBytes: Uint8Array[] = [];
for (let byte = 0; byte < 256; byte++) tokenBytes.push(Uint8Array.from([byte]));
const candidates = [...counts.entries()]
    .filter(([value, count]) => count >= 2 && bytes(value) <= 64)
    .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]));
for (const [value] of candidates.slice(0, 4096 - tokenBytes.length)) {
    tokenBytes.push(new TextEncoder().encode(value));
}

const greedy = new LosslessByteTokenizer(4096);
for (const token of tokenBytes.slice(256)) greedy.add(token);

const configs = [
    { name: "sin contexto", options: { leftContext: 0, rightContext: 0, lookahead: "chars" as const, penalty: 0 } },
    { name: "contexto 2/2 chars", options: { leftContext: 2, rightContext: 2, lookahead: "chars" as const, penalty: 0.3 } },
    { name: "contexto 3/3 palabra", options: { leftContext: 3, rightContext: 3, lookahead: "word" as const, penalty: 0.5 } },
    { name: "contexto 1/1 fuerte", options: { leftContext: 1, rightContext: 1, lookahead: "chars" as const, penalty: 0.7 } },
    { name: "pasado 3, futuro 0", options: { leftContext: 3, rightContext: 0, lookahead: "word" as const, penalty: 0.5 } },
    { name: "pasado 0, futuro 3", options: { leftContext: 0, rightContext: 3, lookahead: "word" as const, penalty: 0.5 } },
];

const ratio = (tokenizer: ContextTokenizer | LosslessByteTokenizer, text: string): number =>
    tokenizer.encode(text).length / [...text].length;

const rows = [];
const greedyRow = {
    name: "byte greedy",
    validation: ratio(greedy, validation),
    external: ratio(greedy, externalText),
    exact: greedy.decode(greedy.encode(validation)) === validation &&
        greedy.decode(greedy.encode(externalText)) === externalText,
};
rows.push(greedyRow);
for (const config of configs) {
    const tokenizer = new ContextTokenizer(tokenBytes, config.options);
    const validationText = tokenizer.decode(tokenizer.encode(validation));
    const externalDecoded = tokenizer.decode(tokenizer.encode(externalText));
    rows.push({
        name: config.name,
        validation: ratio(tokenizer, validation),
        external: ratio(tokenizer, externalText),
        exact: validationText === validation && externalDecoded === externalText,
    });
}

for (const row of rows) {
    console.log(
        `${row.name.padEnd(22)} val ${row.validation.toFixed(3)} · externo ${row.external.toFixed(3)} · exacto ${row.exact ? "sí" : "no"}`,
    );
}

writeFileSync(
    "data/tokenizer-context.json",
    `${JSON.stringify(
        {
            protocol: { vocabulary: tokenBytes.length, maxTokenBytes: 64, train: [...train].length, validation: [...validation].length, external: [...externalText].length },
            rows,
        },
        null,
        2,
    )}\n`,
    "utf8",
);
console.log("guardado: data/tokenizer-context.json");
