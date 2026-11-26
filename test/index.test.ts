import assert from "node:assert/strict";
import { test } from "node:test";

import { VERSION } from "../src/index.ts";

test("la versión publicada coincide con la del paquete", () => {
    assert.equal(VERSION, "0.0.0");
});
