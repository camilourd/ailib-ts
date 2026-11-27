import assert from "node:assert/strict";
import { test } from "node:test";

import { CharTokenizer } from "../../src/data/char.ts";

test("carácter: ida y vuelta con acentos y emoji", () => {
    const tokenizer = new CharTokenizer().train("aña 😀 aña");
    const text = "aña 😀";
    assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    assert.ok(tokenizer.vocabularySize >= 5);
});

test("carácter: lo no visto cae en unk", () => {
    const tokenizer = new CharTokenizer().train("abc");
    assert.deepEqual(tokenizer.encode("aZ"), [tokenizer.encode("a")[0], 0]);
    assert.equal(tokenizer.decode([0]), "\uFFFD");
});
