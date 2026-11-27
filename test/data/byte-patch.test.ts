import assert from "node:assert/strict";
import { test } from "node:test";

import { EntropyByteTokenizer } from "../../src/data/byte-patch.ts";

test("byte-patch: agrupa y reconstruye exactamente", () => {
    const tokenizer = new EntropyByteTokenizer();
    tokenizer.train("hola mundo, hola otra vez. ¡ñandú! 😀", {
        threshold: 2,
        maxPatch: 8,
        vocabularySize: 512,
    });
    for (const text of ["hola mundo", "hola otra vez", "¡ñandú!", "texto no visto 😀"]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("byte-patch: umbral bajo no rompe el decode", () => {
    const tokenizer = new EntropyByteTokenizer();
    tokenizer.train("abc abc abc", { threshold: 0.1, maxPatch: 4, vocabularySize: 300 });
    assert.equal(tokenizer.decode(tokenizer.encode("abc abc")), "abc abc");
});
