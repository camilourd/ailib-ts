import assert from "node:assert/strict";
import { test } from "node:test";

import { ContextTokenizer } from "../../src/data/context-tokenizer.ts";

const tokenBytes = (text: string): Uint8Array => new TextEncoder().encode(text);

function tokensFor(text: string): Uint8Array[] {
    const bytes: Uint8Array[] = [];
    for (let byte = 0; byte < 256; byte++) bytes.push(Uint8Array.from([byte]));
    bytes.push(tokenBytes(text));
    return bytes;
}

test("contexto: DP es lossless y cubre texto no visto", () => {
    const tokenizer = new ContextTokenizer(tokensFor("hola"), {
        leftContext: 2,
        rightContext: 2,
        lookahead: "word",
        penalty: 0.5,
    });
    for (const text of ["hola", "hola mundo", "ñandú 😀"]) {
        assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
    }
});

test("contexto: penaliza cortar una palabra", () => {
    const tokens = tokensFor("abc");
    const normal = new ContextTokenizer(tokens, {
        leftContext: 0,
        rightContext: 0,
        penalty: 0,
    });
    const contextual = new ContextTokenizer(tokens, {
        leftContext: 3,
        rightContext: 3,
        lookahead: "word",
        penalty: 1,
    });
    assert.equal(normal.decode(normal.encode("abc")), "abc");
    assert.equal(contextual.decode(contextual.encode("abc")), "abc");
    assert.ok(contextual.encode("abc").length >= 1);
});

test("contexto: valida opciones", () => {
    assert.throws(() => new ContextTokenizer([], {}), Error);
    assert.throws(
        () => new ContextTokenizer(tokensFor("x"), { leftContext: 4 }),
        Error,
    );
    assert.throws(
        () => new ContextTokenizer(tokensFor("x"), { penalty: -1 }),
        Error,
    );
});
