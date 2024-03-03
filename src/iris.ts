import fs from 'fs';
import Model from "./lib/Model";
import { turn, shuffle } from "./lib/array";

let dataStr = fs.readFileSync('./iris.txt').toString();
let rawData = dataStr.split('\r\n');
let irisType = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'];
let fullData = rawData.map(raw => {
    let data = raw.split(',');
    return { input: data.slice(0, -1).map(d => Number(d)), output: turn(3, irisType.indexOf(data[4])) }
});
fullData = shuffle(fullData);

let model = new Model(4, 210, 29, 24, 3);
model.trains(fullData, .85, 250, (_, a) => a >= 0.99, .82);
model.test(fullData, 35);