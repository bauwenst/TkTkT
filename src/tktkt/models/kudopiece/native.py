"""
Native implementation of KudoPiece segmentation in TkTkT.

TODO: Has not been tested for equivalence to SentencePiece.
"""
from typing import MutableSequence

from ...interfaces.preprocessors import Pretokeniser
from ...interfaces.tokenisers import Preprocessor
from ...models.predictive.viterbi.framework import ViterbiTokeniser, ViterbiObjective, INFTY
from ...models.predictive.viterbi.objectives_guided import ScoreGeneratorUsingSubstringClassifier, SubstringClassifier
from ...models.predictive.viterbi.accumulators import ScoreSum


class UnigramClassifier(SubstringClassifier):

    def __init__(self, log_probabilities: dict[str, float]):
        self.logp = log_probabilities

    def getSegmentLogProbabilities(self, pretoken: str, max_k: int) -> MutableSequence[MutableSequence[float]]:
        return [[self.logp.get(pretoken[n:n+(k+1)], -INFTY) for k in range(max_k)]
                for n in range(len(pretoken))]  # A matrix M where M[n][k] gives the log probability of the occurrence of the token starting at character n with size k+1.


class KudoPieceTokeniser(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, boundaries: Pretokeniser, log_probabilities: dict[str,float]):
        super().__init__(preprocessor, boundaries=boundaries, max_stepsize=max(map(len, log_probabilities)), objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ScoreGeneratorUsingSubstringClassifier(),
                score_combiner=ScoreSum()
            )
        ])
        self.objectives[0].score_generator.setBackend(UnigramClassifier(log_probabilities))
