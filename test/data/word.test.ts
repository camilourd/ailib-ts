import assert from "node:assert/strict";
import { test } from "node:test";

import { WordTokenizer } from "../../src/data/word.ts";

test("palabra: separa palabras, signos y espacios y reconstruye", () => {
    const tokenizer = new WordTokenizer().train("Hola, mundo. Hola, mundo.");
    const text = "Hola, mundo.";
    assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    assert.ok(tokenizer.vocabularySize >= 4);
});

test("palabra: frecuencia mínima y unk", () => {
    const tokenizer = new WordTokenizer().train("raro una vez; común común", { minFrequency: 2 });
    assert.deepEqual(tokenizer.encode("raro"), [0]);
    assert.equal(tokenizer.decode([0]), "\uFFFD");
    assert.throws(() => new WordTokenizer().train("x", { minFrequency: 0 }), Error);
});
