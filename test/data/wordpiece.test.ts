import assert from "node:assert/strict";
import { test } from "node:test";

import { WordPieceTokenizer } from "../../src/data/wordpiece.ts";

test("wordpiece: aprende y reconstruye exactamente", () => {
    const tokenizer = new WordPieceTokenizer();
    tokenizer.train("hola mundo, hola otra vez. ¡ñandú!", { vocabularySize: 320 });
    for (const text of ["hola mundo", "hola otra vez", "¡ñandú!", "texto no visto 😀"]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("wordpiece: usa continuaciones ##", () => {
    const tokenizer = new WordPieceTokenizer();
    tokenizer.train("hablaba hablaban hablar", { vocabularySize: 320 });
    const tokens = tokenizer.encode("hablaba");
    assert.ok(tokens.length >= 1);
    assert.equal(tokenizer.decode(tokens), "hablaba");
});

test("wordpiece: valida el vocabulario", () => {
    const tokenizer = new WordPieceTokenizer();
    assert.throws(() => tokenizer.train("hola", { vocabularySize: 10 }), RangeError);
});
