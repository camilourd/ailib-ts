import assert from "node:assert/strict";
import { test } from "node:test";

import {
    buildTokenizer,
    isLossless,
    programFromGenome,
} from "../../src/data/tokenizer-program.ts";

test("programa: genoma produce una configuración válida", () => {
    const program = programFromGenome([1, 2, 1, 1]);
    assert.equal(program.family, "char");
    assert.equal(program.vocabularySize, 2048);
    assert.equal(program.maxTokenBytes, 64);
    assert.throws(() => programFromGenome([1, 2, 3]), Error);
});

test("programa: diccionario y palabra son lossless sobre texto no visto", () => {
    const train = "hola mundo, hola mundo. ñandú 😀";
    const external = "mundo ñandú nuevo 😀";
    for (const genome of [
        [2, 512, 32, 1],
        [3, 512, 32, 1],
    ]) {
        const tokenizer = buildTokenizer(programFromGenome(genome), train);
        assert.equal(isLossless(tokenizer, [train, external]), true);
        assert.ok(tokenizer.encode(train).length > 0);
    }
});
