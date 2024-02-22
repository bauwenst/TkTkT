import numpy as np
from pathlib import Path
import os

from ...models.viterbi.objectives_guided import CharacterClassifier


def floatToUnicode(f: float) -> str:
    if   f < 0.15:
        return ""
    elif f < 0.29:
        return "▏"
    elif f < 0.43:
        return "▎"
    elif f < 0.57:
        return "▍"
    elif f < 0.71:
        return "▋"
    elif f < 0.85:
        return "▊"
    else:
        return "▉"


def visualisePredictedBoundaries(classifier: CharacterClassifier, word: str):
    logprobs = classifier.getPointLogProbabilities(word)
    probs = np.exp(logprobs)

    result = ""
    separators = list(map(floatToUnicode, probs))
    for character, sep in zip(word, separators):
        result += character + sep

    return result
