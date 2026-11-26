import assert from "node:assert/strict";
import { test } from "node:test";

import { ByteTokenizer } from "../../src/data/tokenizer.ts";

const tokenizer = new ByteTokenizer();

test("ida y vuelta con ASCII, acentos y emoji", () => {
    for (const text of ["hola", "ñandú", "¡Hola, mundo!", "🐢", ""]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("cada token es un byte UTF-8", () => {
    assert.equal(tokenizer.vocabularySize, 256);
    assert.deepEqual(tokenizer.encode("hola"), [104, 111, 108, 97]);
    assert.deepEqual(tokenizer.encode("ñ"), [195, 177]);
    assert.equal(tokenizer.encode("🐢").length, 4);
});

test("decode rechaza tokens que no son bytes", () => {
    assert.throws(() => tokenizer.decode([256]), RangeError);
    assert.throws(() => tokenizer.decode([-1]), RangeError);
    assert.throws(() => tokenizer.decode([1.5]), RangeError);
});
