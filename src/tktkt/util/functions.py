"""
Analytic function prescriptions.
"""
import numpy as np


def relu(x: float) -> float:
    return max(x,0)


def softmax(x: np.ndarray, temperature: float=1.0) -> np.ndarray:
    """
    Shifts to a maximum of 0 first, for numerical stability of the exponential (which is apparently more important
    than the numerical stability of a subtraction).
    https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python

    The temperature has to be applied before this shift, otherwise small temperatures get you in trouble (for inputs matching the sign of the temperature).
    """
    x = x/temperature
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def ln_then_softmax(x: np.ndarray, temperature: float=1.0) -> np.ndarray:
    """Note: only supports positive inputs."""
    return softmax(np.log(x), temperature)


def normalise_then_softmax(x: np.ndarray, temperature: float=1.0) -> np.ndarray:
    return softmax(x / np.sum(x), temperature)


def normalise_then_ln_then_softmax(x: np.ndarray, temperature: float=1.0) -> np.ndarray:
    """Note: only supports positive inputs."""
    return softmax(np.log(x / np.sum(x)), temperature)
