import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";

test("forma, rango, tamaño y strides", () => {
    const tensor = new Tensor(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    assert.equal(tensor.rank, 2);
    assert.equal(tensor.size, 6);
    assert.deepEqual(tensor.shape, [2, 3]);
    assert.deepEqual(tensor.strides, [3, 1]);

    const cube = Tensor.zeros([2, 3, 4]);
    assert.deepEqual(cube.strides, [12, 4, 1]);
});

test("creación: zeros, ones y from", () => {
    assert.deepEqual([...Tensor.zeros([2, 2]).data], [0, 0, 0, 0]);
    assert.deepEqual([...Tensor.ones([2, 2]).data], [1, 1, 1, 1]);
    const tensor = Tensor.from([1, 2, 3]);
    assert.deepEqual(tensor.shape, [3]);
    assert.deepEqual([...tensor.data], [1, 2, 3]);
});

test("un escalar es un tensor de rango 0", () => {
    const scalar = Tensor.from([7], []);
    assert.equal(scalar.rank, 0);
    assert.equal(scalar.size, 1);
    assert.deepEqual(scalar.shape, []);
    assert.equal(scalar.get(), 7);
    scalar.set(8);
    assert.equal(scalar.get(), 8);
});

test("get y set usan los strides", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    assert.equal(tensor.get(0, 0), 1);
    assert.equal(tensor.get(0, 2), 3);
    assert.equal(tensor.get(1, 0), 4);
    assert.equal(tensor.get(1, 2), 6);
    tensor.set(99, 1, 1);
    assert.equal(tensor.get(1, 1), 99);
});

test("índices inválidos fallan", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    assert.throws(() => tensor.get(2, 0), RangeError);
    assert.throws(() => tensor.get(-1, 0), RangeError);
    assert.throws(() => tensor.get(0.5, 0), RangeError);
    assert.throws(() => tensor.get(0), Error);
    assert.throws(() => tensor.set(1, 0, 3), RangeError);
});

test("el constructor valida datos y forma", () => {
    assert.throws(() => new Tensor(new Float32Array(5), [2, 3]), Error);
    assert.throws(() => Tensor.zeros([2, -3]), Error);
    assert.throws(() => Tensor.zeros([2, 1.5]), Error);
});

test("reshape conserva el orden y no comparte datos", () => {
    const original = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const reshaped = original.reshape([3, 2]);
    assert.deepEqual(reshaped.shape, [3, 2]);
    assert.deepEqual([...reshaped.data], [1, 2, 3, 4, 5, 6]);
    assert.equal(reshaped.get(2, 1), 6);
    original.set(99, 0, 0);
    assert.equal(reshaped.get(0, 0), 1);
});

test("reshape de tamaño incompatible falla", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    assert.throws(() => tensor.reshape([4, 2]), Error);
});

test("transpose por defecto invierte los ejes", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    const transposed = tensor.transpose();
    assert.deepEqual(transposed.shape, [3, 2]);
    assert.deepEqual(transposed.strides, [2, 1]);
    assert.deepEqual([...transposed.data], [1, 4, 2, 5, 3, 6]);
    assert.equal(transposed.get(0, 1), 4);
    assert.equal(transposed.get(1, 0), 2);
    assert.equal(transposed.get(2, 1), 6);
});

test("transpose con orden explícito en rango 3", () => {
    const values = Array.from({ length: 24 }, (_, index) => index);
    const tensor = Tensor.from(values, [2, 3, 4]);
    const transposed = tensor.transpose([2, 0, 1]);
    assert.deepEqual(transposed.shape, [4, 2, 3]);
    assert.equal(transposed.get(2, 1, 0), 14);
    const reversed = tensor.transpose();
    assert.deepEqual(reversed.shape, [4, 3, 2]);
});

test("transpose con un orden inválido falla", () => {
    const tensor = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3]);
    assert.throws(() => tensor.transpose([0, 0]), Error);
    assert.throws(() => tensor.transpose([0]), Error);
    assert.throws(() => tensor.transpose([0, 2]), Error);
});
