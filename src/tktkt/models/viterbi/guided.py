from typing import List
from abc import abstractmethod

import numpy as np

from ...interfaces.general import TokeniserWithVocab
from ...preparation.splitters import WordSplitter
from ...preparation.spacemarking import ROBERTA_SPACING


class PointGuidedViterbi(TokeniserWithVocab):  # TODO: Should make use of the general Viterbi framework. You should have one objective that is MaximiseSuggestedSplits() and takes one of these label generators.

    def __init__(self):
        super().__init__(WordSplitter(ROBERTA_SPACING))

    @abstractmethod
    def getLabels(self, pretoken: str) -> List[int]:  # TODO: What if you have probabilities for split points? Could be useful too.
        """
        Should return a binary list which says, for each character, whether it should be followed by a split point or not.
        Whether these splits are feasible, is not the point; they are a heuristic guide.
        """
        pass


from bpe_knockout.project.config import morphologyGenerator
gold_segmentations = {obj.lemma(): obj.morphSplit() for obj in morphologyGenerator()}


class GoldPointsViterbi(PointGuidedViterbi):
    """
    Uses gold segmentations as split point suggestions. This is cheating, but it is a good baseline!
    """

    def getLabels(self, pretoken: str) -> List[int]:
        sow, word = pretoken[0], pretoken[1:]
        if word not in gold_segmentations:
            return [pretoken]

        tokens = gold_segmentations.get(word).split()
        tokens[0] = sow + tokens[0]

        split_positions = np.cumsum([len(t) for t in tokens[:-1]]) - 1  # Alternative for the regex code I normally use. Seems like it'd be faster.
        labels = np.zeros(len(pretoken), dtype=np.int8)
        labels[split_positions] = 1
        return labels.tolist()


class SegmentGuidedViterbi(TokeniserWithVocab):
    pass