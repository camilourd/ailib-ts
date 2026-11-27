// F13: mezcla balanceada de los tres dominios con BPE 2k para el MoE.
import { readFileSync, writeFileSync } from "node:fs";

const domains = ["narrativo", "tecnico", "codigo"] as const;
const trainTokens: number[] = [];
const testSets: Record<string, number[]> = {};
const validationSets: Record<string, number[]> = {};
const bytes: Record<string, number> = {};

for (const domain of domains) {
    const payload = JSON.parse(
        readFileSync(`data/gpu/${domain}-bpe-2k.json`, "utf8"),
    ) as { trainTokens: number[]; testTokens: number[]; validationTokens: number[]; bytes: Record<string, number> };
    trainTokens.push(...payload.trainTokens);
    testSets[domain] = payload.testTokens;
    validationSets[domain] = payload.validationTokens;
    bytes[domain] = payload.bytes.test;
}
bytes.test = bytes.narrativo as number;
bytes.external = bytes.narrativo as number;

writeFileSync(
    "data/gpu/f13-mixture-bpe-2k.json",
    `${JSON.stringify({
        name: "f13-mixture-bpe-2k",
        vocabulary: 2048,
        exact: true,
        trainTokens,
        testTokens: testSets.narrativo,
        externalTokens: validationSets.narrativo,
        testSets,
        validationSets,
        bytes,
    })}\n`,
    "utf8",
);
console.log(`train ${trainTokens.length} tokens; test sets ${Object.keys(testSets).join(", ")}`);
