import numpy as np


from ...models.predictive.viterbi import CharacterClassifier


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


def getPredictionProbabilities(classifier: CharacterClassifier, word: str):
    return np.exp(classifier.getPointLogProbabilities(word))


def visualisePredictedBoundaries(classifier: CharacterClassifier, word: str) -> str:
    probs = getPredictionProbabilities(classifier, word)

    result = ""
    separators = list(map(floatToUnicode, probs))
    for character, sep in zip(word, separators):
        result += character + sep

    return result
