import numpy as np


def softmax(x: np.ndarray, temperature: float=1.0):
    """
    Shifts to a maximum of 0 first, for numerical stability of the exponential (which is apparently more important
    than the numerical stability of a subtraction).
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    exps = np.exp(1/temperature*(x - np.max(x)))
    return exps / np.sum(exps)


def normalise_then_softmax(x: np.ndarray, temperature: float=1.0):
    return softmax(x / np.sum(x), temperature)


def relu(x: float) -> float:
    return max(x,0)
