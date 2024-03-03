import NeuralNetwork from "./NeuralNetwork";
import { dlogistic, logistic } from "./Sigmoid";
import { roundResult, squareArray, turn } from "./array";

export interface Data {
    input: ANN.Inputs;
    output: ANN.Layer.Outputs;
}

interface TrainData {
    data: number;
    test: number;
    acc: number;
}

type stop = (i: number, acc: number) => boolean;

export default class Model {
    model: NeuralNetwork;
    af: ANN.ActivationFunction;
    daf: ANN.ActivationFunction;

    constructor(...units: number[]) {
        let weights = units.slice(0, -1).map((n, i) => squareArray(n, units[i + 1], () => Math.random() / 2 - 0.25));
        let biases = units.slice(1).map(n => new Array(n).fill(0).map(() => Math.random() / 2 - 0.25));
        this.model = new NeuralNetwork([], weights, biases);
        this.af = logistic;
        this.daf = dlogistic;
    }

    setActivationFunction(af: ANN.ActivationFunction, daf: ANN.ActivationFunction) {
        this.af = af;
        this.daf = daf;
        return this;
    }

    raw_train(inputs: ANN.Inputs, trueOutputs: ANN.Layer.Outputs, lr: ANN.LearingRate = 0.9) {
        let output = this.model.output(inputs, this.af);
        let update = this.model.calcUpdateValues(trueOutputs, inputs, lr, this.daf);
        this.model.change(update.weights, update.biases);
        return output;
    }

    train(data: Data[], test: Data[], lr: ANN.LearingRate = 0.9): TrainData {
        let trueTimes = 0;
        data.forEach(d => this.raw_train(d.input, d.output, lr));
        test.forEach(t => {
            let output = this.model.output(t.input);
            if (roundResult(output).join() === t.output.join()) trueTimes++;
        });
        return { data: data.length, test: test.length, acc: trueTimes / test.length };
    }

    trains(fullData: Data[], split: number, times: number, stop: stop = () => false, lr: ANN.LearingRate = 0.9) {
        let sliceIndex = Math.floor(fullData.length * split);
        for (let i = 0; i < times; i++) {
            let { data, test, acc } = this.train(fullData.slice(0, sliceIndex), fullData.slice(sliceIndex, fullData.length), lr);
            console.log(i + 1, '/', times, 'Data:', data, 'Test:', test, 'Acc:', acc);
            if (stop(i, acc)) break;
        }
    }

    test(data: Data[], randomSelect: number | undefined = 0, log: boolean | undefined = true): number {
        let trueTimes = 0;
        if (!randomSelect) {
            data.forEach(d => {
                let o = roundResult(this.model.output(d.input));
                let t = d.output;
                log && console.log('Output:', o, 'True:', t);
                if (o.join() === t.join()) trueTimes++;
            });
        } else for (let i = 0; i < randomSelect; i++) {
            let ri = Math.floor(Math.random() * (data.length));
            let o = roundResult(this.model.output(data[ri].input));
            let t = data[ri].output;
            log && console.log('Output:', o, 'True:', t);
            if (o.join() === t.join()) trueTimes++;
        }
        let acc = trueTimes / (randomSelect || data.length);
        log && console.log('Acc:', acc);
        return acc;
    }
}