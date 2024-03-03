export default class Unit {
    weights: ANN.Unit.Weights;
    bias: ANN.Unit.Bias;

    constructor(w: ANN.Unit.Weights, b: ANN.Unit.Bias) {
        this.weights = w;
        this.bias = b;
    }

    change(w: ANN.Unit.Weights, b: ANN.Unit.Bias) {
        this.weights = w;
        this.bias = b;
    }

    output(i: ANN.Inputs, af: ANN.ActivationFunction): ANN.Unit.Output {
        let sum = 0;
        i.forEach((input, index) => {
            let weight = this.weights[index];
            sum += weight * input;
        });
        sum += this.bias;
        return af(sum);
    }

    calcUpdateValue(errorOutput: number, lastOutputs: ANN.Layer.Outputs, lr: ANN.LearingRate): ANN.Unit.Weights {
        let updateValue: ANN.Unit.Weights = [];
        lastOutputs.forEach((lastOutput, i) => updateValue.push(this.weights[i] + lr * errorOutput * lastOutput));
        return updateValue;
    }
}