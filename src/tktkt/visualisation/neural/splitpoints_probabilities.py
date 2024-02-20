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


def relativePath(path1: Path, path2: Path):
    """
    How do I get from path1 to path2?
    Note: definitely won't work for all path pairs. Just works here.
    """
    result = ""
    for i in range(min(len(path1.parts), len(path2.parts))):
        if path1.parts[i] != path2.parts[i]:
            result += "../"*len(path1.parts[i:])
            result += "/".join(path2.parts[i:])
            break
    return Path(result)


def from_pretrained_absolutePath(cls, absolute_path: Path):
    """
    For some reason, HuggingFace doesn't accept absolute paths for loading models. This is stupid.
    This function fixes that.
    """
    return cls.from_pretrained((relativePath(Path(os.getcwd()), absolute_path)).as_posix())


def visualisePredictedBoundaries(classifier: CharacterClassifier, word: str):
    logprobs = classifier.getPointLogProbabilities(word)
    probs = np.exp(logprobs)

    result = ""
    separators = list(map(floatToUnicode, probs))
    for character, sep in zip(word, separators):
        result += character + sep

    return result
