import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { BPETokenizer } from "../../src/data/bpe.ts";
import { loadBPE, saveBPE } from "../../src/data/bpe-io.ts";

test("BPE: ida y vuelta de merges", (context) => {
    const dir = mkdtempSync(join(tmpdir(), "ailib-bpe-"));
    context.after(() => rmSync(dir, { recursive: true, force: true }));
    const original = new BPETokenizer();
    original.train("baja baja baja, ñandú ñandú", { vocabularySize: 270 });
    const path = join(dir, "bpe.json");
    saveBPE(path, original);
    const loaded = loadBPE(path);
    assert.equal(loaded.vocabularySize, original.vocabularySize);
    assert.deepEqual(loaded.merges, original.merges);
    assert.deepEqual(loaded.encode("baja ñandú"), original.encode("baja ñandú"));
    assert.equal(loaded.decode(loaded.encode("baja ñandú")), "baja ñandú");
});

test("BPE: rechaza archivos inválidos", (context) => {
    const dir = mkdtempSync(join(tmpdir(), "ailib-bpe-"));
    context.after(() => rmSync(dir, { recursive: true, force: true }));
    assert.throws(() => loadBPE(join(dir, "no-existe.json")), Error);
});
