import assert from "node:assert/strict";
import { test } from "node:test";

import { PreTokenizedBPETokenizer } from "../../src/data/pretokenized-bpe.ts";

test("BPE pre-tokenizado es lossless con palabras y números", () => {
    const tokenizer = new PreTokenizedBPETokenizer();
    tokenizer.train("abanico abanico 123 123 abanico", { vocabularySize: 280 });
    const text = "abanico abanico 123, ¡hola!";
    const tokens = tokenizer.encode(text);
    assert.equal(tokenizer.decode(tokens), text);
    assert.ok(tokenizer.vocabularySize >= 256);
});

test("BPE pre-tokenizado separa palabras para no fusionar entre ellas", () => {
    const tokenizer = new PreTokenizedBPETokenizer();
    tokenizer.train("abanico abanico abanico", { vocabularySize: 300 });
    const single = tokenizer.encode("abanico");
    const double = tokenizer.encode("abanico abanico");
    assert.ok(double.length >= single.length * 2);
});
