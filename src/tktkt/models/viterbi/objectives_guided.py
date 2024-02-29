"""
Guided score functions. These are informed by knowledge of the language, e.g. models that estimate the probability of a
token appearing at all (with or without considering context), or models that estimate the probability of a split point.

TODO: Two more ideas:
    - Does the amount of steps you take influence the average score you get? I'm guessing yes, because for example, you
      get equal reward for making one wrong split to get to the end as for making a wrong split, a right split and a
      wrong split. BUT WHAT IF YOU MAKE NO SPLIT? What's the score of using the subword that captures an entire string?
    - Could/should you not do 2*P-1 for the BoundaryAndNonBoundary scores?
    - Also note that there is still quite a big difference between a StringClassifier and the idea of turning a CharacterClassifier
      into a StringClassifier, which is that a StringClassifier normalises across all possible steps from length 1 to K
      that you can take (e.g. a softmax over the vocab, although you lose mass due to the indicator function) whilst if
      you use boundary probabilities, you're only going to be normalised for each of the 2^k boundary configurations of
      a fixed length k.

FIXME: There is something really wacky going on with both of my score grids.
    - The one that "works", possibly isn't stepping to the end of the string, but might be off by 1.
    - The one that doesn't work clearly needs a limit on k.
"""
from typing import List, MutableSequence
from abc import abstractmethod

import numpy as np
import torch

from bpe_knockout.project.config import morphologyGenerator, Pℛ𝒪𝒥ℰ𝒞𝒯

from .framework import ViterbiStepScoreGenerator, ViterbiStepScores, INFTY
from ...preparation.spacemarking import SpaceMarkerLocation
from ...preparation.splitters import WhitespaceAndMarkerPretokeniser


class CharacterClassifier:

    @abstractmethod
    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:  # "MutableSequence" is just "anything that can be indexed with [], has order, and can be modified"
        """
        Returns a list of binary classification log probabilities that say, for each of the n characters, whether it
        should be followed by a split point.

        Note that these are log probabilities, not just probabilities. The reason is two-fold:
            - Neural models produce logits, which need only a constant subtracted to be log probabilities, rather than an expensive softmax.
            - It's more accurate to convert logarithms to probabilities than the inverse due to rounding errors.
              Also, for further computations, it's much more numerically stable to do (-5) + (-5) + (-5) than (10^(-5)) * (10^(-5)) * (10^(-5)).
        """
        pass


class StringClassifier:

    @abstractmethod
    def getSegmentLogProbabilities(self, pretoken: str, max_k: int) -> MutableSequence[MutableSequence[float]]:
        """
        Returns a matrix of log probabilities that say, for each substring of length k starting at each of the n characters,
        what its probability of occurring is across some domain of substrings (e.g. a subword vocabulary).
        """
        pass


#############################################################################################


class BoundaryLikelihood(ViterbiStepScoreGenerator):
    """
    For Viterbi paths that maximise the score on the character boundaries they hit.
    Should be used with sum.
    """

    def __init__(self, point_model: CharacterClassifier):
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
        boundary_scores = [2*np.exp(ln)-1 for ln in self.model.getPointLogProbabilities(string)]  # one entry for each character
        # boundary_scores[-1] = 0  # Score you get from walking to the end is 0. I.e.: it's a good idea, unless you can do better by splitting.

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            K = min(max_k, N-n-1)  # Example: you are at n == 5 and there are N == 7 characters, meaning only n == 6 remains to walk to. Hence, there is K == 1 step available.
            for k in range(K):
                scores.set(n,k, boundary_scores[n+k])  # Example: you are at n == 0. For k == 1, you take the second-smallest step, to 2. The probability of hitting a boundary by doing so is the probability of a boundary being after 1 == n+k.

        return scores


class BoundaryAndNonBoundaryLikelihood(ViterbiStepScoreGenerator):
    """
    Uses point boundaries, assumed to be independent, to compute segment probabilities.
    Should be used with sum accumulation since scores are in log space.

    FIXME: It's actually not obvious if summing is appropriate, because it could be that in log space, being 40% sure
           of a boundary results in a much bigger punishment than the reward of being 60% sure of a boundary.
    """

    def __init__(self, point_model: CharacterClassifier):
        self.model = point_model

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        """
        Let's say you have suggested splits abc|de|f. The probability of the segment "abc" is computed now not as
            P(boundary after char 2 | abcdef)
        but instead as
            P(boundary after char 2 | abcdef)*P(no boundary after char 1 | abcdef)*P(no boundary after char 0 | abcdef)

        Computations are done in log space and with addition, to keep numerical stability.

        There's a funky equivalence here: you could convert the CharacterClassifier into a StringClassifier first, and
        then use that StringClassifier with its usual grid generator.
        """
        boundary_log_probabilities = self.model.getPointLogProbabilities(string)  # one entry for each character
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            for k in range(max_k):
                log_probability = boundary_log_probabilities[n+k]  # This is like in the previous class.
                for i in range(k):
                    logp = boundary_log_probabilities[n+i]
                    log_probability += np.log(1-np.exp(logp))  # ln(1-p) == ln(1 - e^(ln p))
                scores.set(n, k, log_probability)

        return scores


###########################################################################################


class GoldSplits(CharacterClassifier):
    """
    Uses gold segmentations as split point suggestions. This is cheating, but it is a good baseline!

    TODO: Has to be able to handle byte-based input too. What you receive for scoring is a pretoken about to be
          tokenised. That means for ë you will receive byte-based input and won't find it in your gold dictionary.
          preprocessor.undo() probably works.
    """

    def __init__(self, pretokeniser: WhitespaceAndMarkerPretokeniser):
        self.pretokeniser = pretokeniser
        self.pretoken_shift = len(self.pretokeniser.marker.substitute)*(self.pretokeniser.marker.location == SpaceMarkerLocation.START)
        self.gold_segmentations = {obj.lemma(): obj.morphSplit() for obj in morphologyGenerator()}

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        labels = np.zeros(len(pretoken), dtype=np.float32)

        word, _ = self.pretokeniser.stripMarker(pretoken)
        if word in self.gold_segmentations:
            tokens = self.gold_segmentations[word].split()
            split_positions = np.cumsum([len(t) for t in tokens[:-1]]) - 1
            split_positions += self.pretoken_shift  # If "word" is shown to the character model as "Ġword", the suggested split indices should shift by 1.

            labels[split_positions] = 1

        return np.log(labels)


from transformers.models.canine.modeling_canine import CanineForTokenClassification, TokenClassifierOutput
from transformers.models.canine.tokenization_canine import CanineTokenizer

class HuggingFaceCharacterModelForTokenClassification(CharacterClassifier):
    """
    NOTE: Outputs log probabilities, not probabilities.
    """

    def __init__(self, characters_to_modelinput: CanineTokenizer, for_token_classification: CanineForTokenClassification,  # Sadly there is no generic "ForTokenClassification" type in HuggingFace's API, but any should work.
                 input_kwargs: dict=None):
        self.input_generator = characters_to_modelinput
        self.model           = for_token_classification
        self.generator_args = input_kwargs or dict()

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        input_to_model = self.input_generator(pretoken, add_special_tokens=False, return_tensors="pt", **self.generator_args)
        with torch.no_grad():  # no_grad means all tensors returned don't have their gradient tracked, so you don't need to .detach() them before going to numpy.
            prediction: TokenClassifierOutput = self.model(**input_to_model)

        chars_by_classes = prediction.logits.squeeze()  # Remove batch dimension (because it has size 1).
        normalisation_constants = torch.logsumexp(chars_by_classes, dim=1)  # Note that normalisation happens not OVER characters, but PER character. It happens over two binary classes, N times.
        positive_logits = chars_by_classes[:,1]
        logprobabilities = positive_logits - normalisation_constants
        return logprobabilities.numpy()
