import numpy as np


def softmax(x: np.ndarray):
    """
    Shifts to a maximum of 0 first, for numerical stability of the exponential (which is apparently more important
    than the numerical stability of a subtraction).
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    """
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def relu(x: float) -> float:
    return max(x,0)
