import numpy as np
from scipy.stats import entropy


def shannon(probabilities: np.ndarray):
    return entropy(pk=probabilities, base=2)


def renyi(probabilities: np.ndarray, alpha: float=2.0):
    """
    Computes Rényi entropy, a generalisation of Shannon entropy.
    https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

    Shannon entropy appears for alpha = 1.0, which causes L'Hôpital's rule to be applied and turns log(sum) into sum(log).

    Sometimes the case alpha = 2.0 is called "the" Rényi entropy, equivalent to the negative log probability of two
    tokens being equal if both are sampled at random. However, to be fully correct, you should always use "Rényi at alpha = ...".
    https://aclanthology.org/2023.acl-long.284v2.pdf
    """
    if alpha == 1.0:
        return shannon(probabilities)

    return 1/(1-alpha)*np.log2(np.sum(probabilities ** alpha))