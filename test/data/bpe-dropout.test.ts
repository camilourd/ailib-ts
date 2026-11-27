import assert from "node:assert/strict";
import { test } from "node:test";

import { BPEDropoutTokenizer } from "../../src/data/bpe-dropout.ts";

test("BPE dropout es lossless con y sin dropout", () => {
    const tokenizer = new BPEDropoutTokenizer({ rate: 0.2, seed: 1 });
    tokenizer.train("abanico abanico abanico abanico", { vocabularySize: 260 });
    const text = "abanico abanico";
    const normal = tokenizer.encode(text);
    const dropped = tokenizer.encode(text, { dropout: true });
    assert.equal(tokenizer.decode(normal), text);
    assert.equal(tokenizer.decode(dropped), text);
});

test("BPE dropout valida la tasa", () => {
    assert.throws(() => new BPEDropoutTokenizer({ rate: 1 }), RangeError);
    assert.throws(() => new BPEDropoutTokenizer({ rate: -0.1 }), RangeError);
});
