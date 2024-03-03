export const tanh: ANN.ActivationFunction = Math.tanh;
export const dtanh: ANN.ActivationFunction = x => 1 / (Math.cosh(Math.atanh(x)) ** 2);
export const logistic: ANN.ActivationFunction = x => 1 / (1 + Math.E ** (-x));
export const dlogistic: ANN.ActivationFunction = x => x * (1 - x);