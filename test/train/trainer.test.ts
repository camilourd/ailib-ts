import assert from "node:assert/strict";
import { test } from "node:test";

import { Tensor } from "../../src/core/tensor.ts";
import { mean, mul, relu } from "../../src/core/tensor-ops.ts";
import { crossEntropy } from "../../src/nn/losses.ts";
import { Linear } from "../../src/nn/linear.ts";
import { SGD } from "../../src/optim/sgd.ts";
import { train } from "../../src/train/trainer.ts";

test("entrena XOR con una capa oculta", () => {
    const inputs = Tensor.from([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]);
    const targets = [0, 1, 1, 0];
    const hidden = new Linear(2, 4, { seed: 1 });
    const output = new Linear(4, 2, { seed: 2 });
    const optimizer = new SGD([...hidden.parameters(), ...output.parameters()], 0.5);

    const history = train(
        (input) => output.forward(relu(hidden.forward(input))),
        inputs,
        {
            steps: 3000,
            optimizer,
            loss: (prediction) => crossEntropy(prediction, targets),
        },
    );

    assert.equal(history.length, 3000);
    const finalLoss = history.at(-1) as number;
    assert.ok(finalLoss < 0.02, `pérdida final ${finalLoss}`);

    const logits = output.forward(relu(hidden.forward(inputs)));
    let correct = 0;
    for (let row = 0; row < 4; row++) {
        const predicted = logits.get(row, 1) > logits.get(row, 0) ? 1 : 0;
        if (predicted === targets[row]) correct += 1;
    }
    assert.equal(correct, 4);
});

test("train valida pasos y pérdida escalar", () => {
    const optimizer = new SGD([], 0.1);
    const input = Tensor.from([1], [1]);
    assert.deepEqual(train((value) => value, input, { steps: 0, optimizer, loss: (p) => p }), []);
    assert.throws(
        () => train((value) => value, input, { steps: 1, optimizer, loss: (p) => p }),
        Error,
    );
});

test("onStep recibe el paso y la pérdida", () => {
    const parameter = Tensor.from([1], [1], { requiresGrad: true });
    const optimizer = new SGD([parameter], 0.1);
    const seen: Array<[number, number]> = [];
    const history = train(
        (value) => value,
        Tensor.from([0], [1]),
        {
            steps: 2,
            optimizer,
            loss: () => mean(mul(parameter, parameter)),
            onStep: (step, loss) => seen.push([step, loss]),
        },
    );
    assert.equal(history.length, 2);
    assert.deepEqual(seen.map(([step]) => step), [0, 1]);
    assert.ok((history[0] as number) > (history[1] as number));
});
