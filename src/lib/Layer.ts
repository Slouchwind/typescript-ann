import { dlogistic } from "./Sigmoid";
import Unit from "./Unit";

export default class Layer {
    weights: ANN.Layer.Weights;
    biases: ANN.Layer.Biases;
    Units: Unit[];

    constructor(w: ANN.Layer.Weights, b: ANN.Layer.Biases) {
        this.weights = w;
        this.biases = b;
        this.Units = [];
        w.forEach((weights, index) => {
            this.Units.push(new Unit(weights, b[index]));
        });
    }

    change(w: ANN.Layer.Weights, b: ANN.Layer.Biases) {
        this.weights = w;
        this.biases = b;
        this.Units.forEach((unit, i) => unit.change(w[i], b[i]));
    }

    output(i: ANN.Inputs, af: ANN.ActivationFunction): ANN.Layer.Outputs {
        // console.log({ i, w: this.weights, b: this.biases });
        let output: ANN.Layer.Outputs = [];
        this.Units.forEach(unit => {
            output.push(unit.output(i, af));
        });
        return output;
    }

    calcOutputError(
        outputs: ANN.Layer.Outputs, trueOutputs: ANN.Layer.Outputs,
        daf: ANN.ActivationFunction = dlogistic
    ): ANN.Layer.ErrorOutputs {
        let nodeError: ANN.Layer.ErrorOutputs[0] = [];
        outputs.forEach((output, index) => {
            nodeError.push(daf(output) * (trueOutputs[index] - output));
        });
        return [nodeError, this.weights];
    }

    calcError(
        outputs: ANN.Layer.Outputs,
        lastError: ANN.Layer.Outputs, lastWeights: ANN.Layer.Weights,
        daf: ANN.ActivationFunction = dlogistic
    ): ANN.Layer.ErrorOutputs {
        let nodeError: ANN.Layer.ErrorOutputs[0] = [];
        outputs.forEach((output, index) => {
            let sum = 0;
            lastError.forEach((err, i) => {
                let weight = lastWeights[i][index];
                sum += err * weight;
            });
            nodeError.push(daf(output) * sum);
        })
        return [nodeError, this.weights];
    }

    calcUpdateValues(
        errorOutputs: ANN.Layer.ErrorOutputs[0], lastLayerOutput: ANN.Layer.Outputs,
        lr: ANN.LearingRate
    ): ANN.Layer.UpdateOutputs {
        let updateWeightValues: ANN.Layer.Weights = [];
        let updateBiasValues: ANN.Layer.Biases = [];
        this.Units.forEach((unit, index) => {
            updateWeightValues.push(unit.calcUpdateValue(errorOutputs[index], lastLayerOutput, lr));
            updateBiasValues.push(this.biases[index] + lr * errorOutputs[index]);
        });
        return { weights: updateWeightValues, biases: updateBiasValues };
    }
}