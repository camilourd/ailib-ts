import assert from "node:assert/strict";
import { test } from "node:test";

import { SentencePieceTokenizer } from "../../src/data/sentencepiece.ts";

test("sentencepiece: flujo crudo y espacios", () => {
    const tokenizer = new SentencePieceTokenizer();
    tokenizer.train("hola mundo, hola otra vez. ¡ñandú!", { vocabularySize: 512 });
    for (const text of ["hola mundo", "hola  otra vez", "sin espacios", "¡ñandú! 😀"]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("sentencepiece: marcador literal escapado", () => {
    const tokenizer = new SentencePieceTokenizer();
    tokenizer.train("texto ▁ con marcador", { vocabularySize: 512 });
    const text = "texto ▁ con marcador";
    assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
});
