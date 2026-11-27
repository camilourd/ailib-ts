// F12.4b: efecto de BPE-dropout en tokens/carácter.
import { readFileSync, writeFileSync } from "node:fs";

import { BPEDropoutTokenizer } from "../../src/data/bpe-dropout.ts";

const rows: Array<Record<string, number | string | boolean>> = [];
for (const domain of ["narrativo", "tecnico", "codigo"] as const) {
    const train = readFileSync(`data/corpora/${domain}-train.txt`, "utf8").slice(0, 50000);
    const test = readFileSync(`data/corpora/${domain}-test.txt`, "utf8");
    const tokenizer = new BPEDropoutTokenizer({ rate: 0.1, seed: 1 });
    tokenizer.train(train, { vocabularySize: 2048 });
    const normal = tokenizer.encode(test);
    const dropped = tokenizer.encode(test, { dropout: true });
    rows.push({
        domain,
        vocabulary: tokenizer.vocabularySize,
        normal: normal.length / [...test].length,
        dropout: dropped.length / [...test].length,
        exact: tokenizer.decode(dropped) === test,
    });
    console.log(`${domain}: normal ${(normal.length / [...test].length).toFixed(3)} · dropout ${(dropped.length / [...test].length).toFixed(3)}`);
}
writeFileSync("data/tokenizer-bpe-dropout.json", `${JSON.stringify(rows, null, 2)}\n`, "utf8");
