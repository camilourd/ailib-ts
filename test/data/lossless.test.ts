import assert from "node:assert/strict";
import { test } from "node:test";

import { LosslessByteTokenizer } from "../../src/data/lossless.ts";

test("lossless: los bytes cubren cualquier texto", () => {
    const tokenizer = new LosslessByteTokenizer(300);
    for (const text of ["hola", "ñandú 😀", "línea\nnueva"]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("lossless: un token compuesto acorta y se reconstruye", () => {
    const tokenizer = new LosslessByteTokenizer(300);
    const id = tokenizer.add(new TextEncoder().encode("ab"));
    assert.notEqual(id, null);
    const tokens = tokenizer.encode("abab");
    assert.ok(tokens.length < 4);
    assert.equal(tokenizer.decode(tokens), "abab");
});

test("lossless: valida el límite y los tokens", () => {
    assert.throws(() => new LosslessByteTokenizer(255), Error);
    const tokenizer = new LosslessByteTokenizer(256);
    assert.equal(tokenizer.add(new TextEncoder().encode("xy")), null);
    assert.throws(() => tokenizer.decode([999]), RangeError);
});
