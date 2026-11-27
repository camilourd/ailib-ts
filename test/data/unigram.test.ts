import assert from "node:assert/strict";
import { test } from "node:test";

import { UnigramTokenizer } from "../../src/data/unigram.ts";

test("unigram: EM y Viterbi son lossless", () => {
    const tokenizer = new UnigramTokenizer();
    tokenizer.train("hola mundo, hola otra vez. ¡ñandú!", {
        vocabularySize: 400,
        iterations: 2,
        minFrequency: 1,
    });
    for (const text of ["hola mundo", "hola otra vez", "¡ñandú!", "texto no visto 😀"]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("unigram: aprende piezas multi-carácter", () => {
    const tokenizer = new UnigramTokenizer();
    tokenizer.train("hablaba hablaban hablar hablaba", {
        vocabularySize: 400,
        iterations: 2,
        minFrequency: 1,
    });
    const tokens = tokenizer.encode("hablaba");
    assert.ok(tokens.length >= 1);
    assert.equal(tokenizer.decode(tokens), "hablaba");
});

test("unigram: valida opciones", () => {
    const tokenizer = new UnigramTokenizer();
    assert.throws(() => tokenizer.train("hola", { vocabularySize: 10 }), RangeError);
    assert.throws(() => tokenizer.train("hola", { maxTokenLength: 0 }), RangeError);
});
