import numpy as np

# def sigmoid(x):
#     return 1/(1+np.exp(-x))

class Sigmoid():
    @staticmethod
    def get_value(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def get_derivative(x):
        return Sigmoid.get_value(x) - Sigmoid.get_value(x)**2


class TanH:
    @staticmethod
    def get_value(x):
        return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

    @staticmethod
    def get_derivative(x):
        return 1 - TanH().get_value(x) ** 2


class ReLU():
    @staticmethod
    def get_value(x):
        return np.maximum(0, x)

    @staticmethod
    def get_derivative(x):
        return np.where(x <= 0, 0, 1)


class Identity():
    @staticmethod
    def get_value(x):
        return np.ones(x.shape) * x

    @staticmethod
    def get_derivative(x):
        return np.ones(x.shape)


class Softmax():
    @staticmethod
    def get_value(x):
        return np.exp(x)/np.sum(np.exp(x), axis=1).reshape(-1, 1)