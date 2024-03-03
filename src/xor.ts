import Model, { Data } from "./lib/Model";

let fullData: Data[] = [
    { input: [1, 1], output: [1, 0] },
    { input: [1, 0], output: [0, 1] },
    { input: [0, 1], output: [0, 1] },
    { input: [0, 0], output: [1, 0] },
];

let model = new Model(2, 20, 30, 2);
model.trains(fullData, 1, 100);
model.test(fullData);