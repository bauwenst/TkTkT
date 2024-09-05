"""
Batch computations on fixed-length arrays.
"""
from abc import ABC, abstractmethod
import numpy as np

from .functions import *

__all__ = ["BatchNormalisation", "IdentityBatchNormalisation", "LinearNormalisation", "SoftmaxNormalisation", "PowerNormalisation"]


class BatchNormalisation(ABC):

    @abstractmethod
    def normalise(self, values: np.ndarray) -> np.ndarray:
        pass


class IdentityBatchNormalisation(BatchNormalisation):

    def normalise(self, values: np.ndarray) -> np.ndarray:
        return values


class LinearNormalisation(BatchNormalisation):

    def normalise(self, values: np.ndarray) -> np.ndarray:
        return values / np.sum(values)  # Faster than Pythonic sum() for Numpy arrays: https://stackoverflow.com/q/10922231/9352077


class SoftmaxNormalisation(BatchNormalisation):

    def __init__(self, temperature: float=1.0, normalise_linearly_first: bool=False):
        self.tau = temperature
        self.do_scale = normalise_linearly_first

    def normalise(self, values: np.ndarray) -> np.ndarray:
        if self.do_scale:
            return normalise_then_softmax(values, temperature=self.tau)
        else:
            return softmax(values, temperature=self.tau)


class PowerNormalisation(BatchNormalisation):
    """
    Raise every element to a power p and then do linear normalisation. This is equivalent to taking a logarithm and
    then a softmax with temperature 1/p, because x_i^p/\sum_j x_j^p == e^{p\ln x_i}/\sum_j e^{p\ln x_j}.

    At a temperature of 1.0, p = 1 and hence this reduces to linear normalisation. Softmax has no temperature where
    this is the case.
    """

    def __init__(self, temperature: float=1.0, normalise_linearly_first: bool=False):
        self.tau = temperature
        self.power = 1/temperature
        self.do_scale = normalise_linearly_first

    def normalise(self, values: np.ndarray) -> np.ndarray:
        if self.do_scale:
            return normalise_then_ln_then_softmax(values, temperature=self.tau)
        else:
            return ln_then_softmax(values, temperature=self.tau)
