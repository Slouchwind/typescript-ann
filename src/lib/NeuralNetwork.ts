import Layer from "./Layer";
import { dlogistic, logistic } from "./Sigmoid";

export default class NeuralNetwork {
    unitOutputs: ANN.NeuralNetwork.UnitOutputs;
    weights: ANN.NeuralNetwork.Weights;
    biases: ANN.NeuralNetwork.Biases;
    Layers: Layer[];

    constructor(
        u: ANN.NeuralNetwork.UnitOutputs, w: ANN.NeuralNetwork.Weights, b: ANN.NeuralNetwork.Biases
    ) {
        this.unitOutputs = u;
        this.weights = w;
        this.biases = b;
        this.Layers = [];
        w.forEach((layerWeights, index) => {
            this.Layers.push(new Layer(layerWeights, b[index]));
        });
    }

    change(w: ANN.NeuralNetwork.Weights, b: ANN.NeuralNetwork.Biases) {
        this.weights = w;
        this.biases = b;
        this.Layers.forEach((layer, i) => layer.change(w[i], b[i]));
    }

    output(i: ANN.Inputs, af: ANN.ActivationFunction | undefined = logistic): ANN.NeuralNetwork.Outputs {
        let layerOutputs: ANN.Layer.Outputs = i;
        this.Layers.forEach((layer, index) => {
            layerOutputs = layer.output(layerOutputs, af);
            this.unitOutputs[index] = layerOutputs;
        });
        return layerOutputs;
    }

    calcErrors(trueOutputs: ANN.Layer.Outputs, daf: ANN.ActivationFunction = dlogistic): ANN.NeuralNetwork.ErrorOutputs {
        let lastErrorOutputs: ANN.Layer.ErrorOutputs = [[], []];
        let errorOutputs: ANN.NeuralNetwork.ErrorOutputs = [];
        this.unitOutputs.reverse().forEach((outputs, rindex) => {
            let index = this.unitOutputs.length - 1 - rindex;
            if (lastErrorOutputs[0][0] === undefined) {
                lastErrorOutputs = this.Layers[index].calcOutputError(outputs, trueOutputs, daf);
            } else {
                lastErrorOutputs = this.Layers[index].calcError(outputs, ...lastErrorOutputs, daf);
            }
            // console.log({ lastErrorOutputs });
            errorOutputs.push(lastErrorOutputs[0]);
        });
        this.unitOutputs.reverse();
        return errorOutputs.reverse();
    }

    calcUpdateValues(
        trueOutputs: ANN.Layer.Outputs, i: ANN.Inputs,
        lr: ANN.LearingRate = 0.9,
        daf: ANN.ActivationFunction = dlogistic
    ): ANN.NeuralNetwork.UpdateOutputs {
        let errorOutputs = this.calcErrors(trueOutputs, daf);
        // console.log(errorOutputs);
        let updateWeightValues: ANN.NeuralNetwork.Weights = [];
        let updateBiasValues: ANN.NeuralNetwork.Biases = [];
        this.Layers.forEach((layer, index) => {
            let layerUpdateValues = layer.calcUpdateValues(errorOutputs[index], this.unitOutputs[index - 1] || i, lr);
            // console.log(layerUpdateValues);
            updateWeightValues.push(layerUpdateValues.weights);
            updateBiasValues.push(layerUpdateValues.biases);
        });
        return { weights: updateWeightValues, biases: updateBiasValues };
    }
}