"""
Guided score functions. These are informed by knowledge of the language, e.g. models that estimate the probability of a
token appearing at all (with or without considering context), or models that estimate the probability of a split point.

TODO: Now make these compatible with a subword vocabulary... I have a feeling that I should implement this as a
      general post-processing step, e.g. in a superclass between ViterbiStepScoreGenerator and the classes below.
      It's not really that difficult: any step that isn't in a given vocab, gets zero probability or -INFTY likelihood.
      |
      Actually, for the segment classifier class, it's likely that the class itself will handle this, because it just
      doesn't even have a vague suggestion for the probability of a non-existing subword, and they don't normalise over
      the domain of all strings but only the existing subwords.
      |
      For the point classifiers, it likely also isn't a problem, because there the probabilities are all in the binary
      domain. A split point has a baseline probability that is collected by any step that reaches it, UNLESS that step
      doesn't exist in the subword vocab. Indeed, the same split point now has two probabilities associated with it
      rather than one: its actual probability, and 0, depending on the step. Works out fine!
"""
from typing import List
from abc import abstractmethod

import numpy as np
import torch

from bpe_knockout.project.config import morphologyGenerator

from .framework import ViterbiStepScoreGenerator, ViterbiStepScores, INFTY


class SplitpointClassifier:

    @abstractmethod
    def getPointProbabilities(self, pretoken: str) -> List[float]:
        """
        Returns a list of binary classification probabilities that say, for each of the n characters, whether it
        should be followed by a split point.
        """
        pass


class StringClassifier:

    @abstractmethod
    def getSegmentProbabilities(self, pretoken: str, max_k: int) -> List[List[float]]:
        """
        Returns a matrix of probabilities that say, for each substring of length k starting at each of the n characters,
        what its probability of occurring is across some domain of substrings (e.g. a subword vocabulary).
        """
        pass


#############################################################################################


class MaximiseSplitsOnBoundaries(ViterbiStepScoreGenerator):
    """
    For Viterbi paths that maximise the score on the character boundaries they hit.
    Should be used with sum.
    """

    def __init__(self, point_model: SplitpointClassifier):
        self.model = point_model

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        """
        Point models give a probability of each character having a split point after it or not.

        Here we want a Viterbi path to accumulate as much of this probability mass as possible (hit all the predicted
        boundaries), but disincentivise cheating by oversegmenting (which would, indeed, collect all boundary
        probability, without being punished for collecting 0s).

        Hence, the probabilities are rescaled by 2*P - 1: you are rewarded for stepping on a boundary and you are
        punished for stepping on a non-boundary. I suspect this will need a tiebreaker objective.

        The alternative is to use multiplication, but this gives outsized punishment to stepping on a non-boundary (a
        single non-boundary can make 20 boundary hits evaporate by multiplying by a very small number). This is also
        easy to see when you take the ln, where you are now punished by -INFTY if you get it wrong but getting it right
        is invisible.
        """
        boundary_scores = [2*label-1 for label in self.model.getPointProbabilities(string)]  # one entry for each character
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            K = min(max_k, N-n-1)  # Example: you are at n == 5 and there are N == 7 characters, meaning only n == 6 remains to walk to. Hence, there is K == 1 step available.
            for k in range(K):
                scores.set(n,k, boundary_scores[n+k])  # Example: you are at n == 0. For k == 1, you take the second-smallest step, to 2. The probability of hitting a boundary by doing so is the probability of a boundary being after 1 == n+k.

        return scores


class MaximiseSplitsEverywhere(ViterbiStepScoreGenerator):
    """
    Uses point boundaries, assumed to be independent, to compute segment probabilities.
    Should be used with product, or be ln'd and be used with sum.
    """

    def __init__(self, point_model: SplitpointClassifier):
        self.model = point_model

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        """
        Let's say you have suggested splits abc|de|f. The probability of the segment "abc" is computed now not as
            P(boundary after char 2 | abcdef)
        but instead as
            P(boundary after char 2 | abcdef)*P(no boundary after char 1 | abcdef)*P(no boundary after char 0 | abcdef)
        """
        boundary_probabilities = self.model.getPointProbabilities(string)  # one entry for each character
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            for k in range(max_k):
                probability = boundary_probabilities[n+k]  # This is like the above class.
                for i in range(k):
                    probability *= (1-boundary_probabilities[n+i])
                scores.set(n, k, probability)

        return scores


###########################################################################################


class GoldSplits(SplitpointClassifier):
    """
    Uses gold segmentations as split point suggestions. This is cheating, but it is a good baseline!
    """

    def __init__(self):
        self.gold_segmentations = {obj.lemma(): obj.morphSplit() for obj in morphologyGenerator()}

    def getLabels(self, pretoken: str) -> List[float]:
        labels = np.zeros(len(pretoken), dtype=np.int8)

        sow, word = pretoken[0], pretoken[1:]  # TODO: Obviously this is heresy.
        if word in self.gold_segmentations:
            tokens = self.gold_segmentations[word].split()
            tokens[0] = sow + tokens[0]

            split_positions = np.cumsum([len(t) for t in tokens[:-1]]) - 1  # Alternative for the regex code I normally use. Seems like it'd be faster.
            labels[split_positions] = 1

        return labels.tolist()


from transformers.models.canine.modeling_canine import CanineForTokenClassification, TokenClassifierOutput
from transformers.models.canine.tokenization_canine import CanineTokenizer

class HuggingFaceCharacterModelForTokenClassification(SplitpointClassifier):

    def __init__(self, characters_to_model_input: CanineTokenizer, for_token_classification: CanineForTokenClassification):  # Sadly there is no generic "ForTokenClassification" type in HuggingFace's API, but any should work.
        self.input_generator = characters_to_model_input
        self.model           = for_token_classification

    def getPointProbabilities(self, pretoken: str) -> List[float]:
        input_to_model = self.input_generator(pretoken, add_special_tokens=False, return_tensors="pt")
        prediction: TokenClassifierOutput = self.model(**input_to_model)

        chars_by_classes = prediction.logits.squeeze()  # Remove batch dimension (because it has size 1).
        positive_logits = chars_by_classes[:,1]  # Logits are equal to log probabilities up to a constant, so you actually don't need to argmax if you want the likelihood.
        normalisation_constant = torch.logsumexp(positive_logits, dim=0)
        logprobabilities = positive_logits #- normalisation_constant   # You actually don't need to normalise for a Viterbi trellis, because you're only comparing anyway.
        return logprobabilities.tolist()
