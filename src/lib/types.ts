declare namespace ANN {
    type Inputs = number[];
    type ActivationFunction = (n: number) => number;
    type LearingRate = number;

    namespace Unit {
        type Weights = number[];
        type Bias = number;
        type Output = number;
    }

    namespace Layer {
        type Weights = Unit.Weights[];
        type Biases = Unit.Bias[];
        type Outputs = Unit.Output[];
        type ErrorOutputs = [Outputs, Weights];
        interface UpdateOutputs {
            weights: Weights;
            biases: Biases;
        }
    }

    namespace NeuralNetwork {
        type UnitOutputs = Layer.Outputs[];
        type Weights = Layer.Weights[];
        type Biases = Layer.Biases[];
        type Outputs = Layer.Outputs;
        type ErrorOutputs = Layer.ErrorOutputs[0][];
        interface UpdateOutputs {
            weights: Weights;
            biases: Biases;
        }
    }
}