import assert from "node:assert/strict";
import { test } from "node:test";

import { BPETokenizer } from "../../src/data/bpe.ts";

test("BPE aprende merges y reduce tokens", () => {
    const bpe = new BPETokenizer();
    bpe.train("baja baja baja", { vocabularySize: 260, minFrequency: 2 });
    assert.ok(bpe.vocabularySize > 256);
    const tokens = bpe.encode("baja baja baja");
    assert.ok(tokens.length < 15, `tokens ${tokens.length}`);
    assert.equal(bpe.decode(tokens), "baja baja baja");
});

test("BPE conserva acentos y emoji en la ida y vuelta", () => {
    const bpe = new BPETokenizer();
    bpe.train("ñandú ñandú 🐢 🐢", { vocabularySize: 280 });
    const text = "ñandú 🐢";
    assert.equal(bpe.decode(bpe.encode(text)), text);
});

test("la misma entrada produce los mismos merges", () => {
    const first = new BPETokenizer();
    const second = new BPETokenizer();
    first.train("abc abc abc", { vocabularySize: 260 });
    second.train("abc abc abc", { vocabularySize: 260 });
    assert.deepEqual(first.merges, second.merges);
    assert.deepEqual(first.encode("abc"), second.encode("abc"));
});

test("BPE valida tokens y parámetros", () => {
    const bpe = new BPETokenizer();
    assert.throws(() => bpe.decode([256]), RangeError);
    assert.throws(() => bpe.decode([-1]), RangeError);
    assert.throws(() => bpe.train("x", { vocabularySize: 255 }), RangeError);
    assert.throws(() => bpe.train("x", { minFrequency: 0 }), RangeError);
});
