// Teoría: docs/teoria/nucleo/bpe.md

import { readFileSync, renameSync, writeFileSync } from "node:fs";

import { BPETokenizer } from "./bpe.ts";
import type { Merge } from "./bpe.ts";

export const BPE_FORMAT = "ailib-ts.bpe";
export const BPE_VERSION = 1;

/** Guarda los merges de un tokenizer BPE en JSON versionado. */
export function saveBPE(path: string, tokenizer: BPETokenizer): void {
    const payload = {
        format: BPE_FORMAT,
        version: BPE_VERSION,
        merges: tokenizer.merges,
    };
    const temporary = `${path}.tmp`;
    writeFileSync(temporary, `${JSON.stringify(payload)}\n`, "utf8");
    renameSync(temporary, path);
}

/** Carga un tokenizer BPE desde un JSON guardado con `saveBPE`. */
export function loadBPE(path: string): BPETokenizer {
    let parsed: unknown;
    try {
        parsed = JSON.parse(readFileSync(path, "utf8"));
    } catch {
        throw new Error(`no se puede leer el tokenizer BPE ${path}`);
    }
    if (typeof parsed !== "object" || parsed === null) {
        throw new Error(`${path}: BPE inválido`);
    }
    const record = parsed as Record<string, unknown>;
    if (record.format !== BPE_FORMAT || record.version !== BPE_VERSION) {
        throw new Error(`${path}: formato o versión de BPE inválidos`);
    }
    if (!Array.isArray(record.merges)) throw new Error(`${path}: faltan los merges`);
    const tokenizer = new BPETokenizer();
    tokenizer.restore(record.merges as Merge[]);
    return tokenizer;
}
