"""
Native implementation of KudoPiece segmentation in TkTkT.

TODO: Has not been tested for equivalence to SentencePiece.
"""
from typing import Dict, MutableSequence

from ...interfaces.tokenisers import Preprocessor
from ...models.predictive.viterbi.framework import ViterbiTokeniser, ViterbiObjective, INFTY
from ...models.predictive.viterbi import ScoreGeneratorUsingSubstringClassifier, SubstringClassifier
from ...models.predictive.viterbi.accumulators import ScoreSum


class UnigramClassifier(SubstringClassifier):

    def __init__(self, log_probabilities: Dict[str, float]):
        self.logp = log_probabilities

    def getSegmentLogProbabilities(self, pretoken: str, max_k: int) -> MutableSequence[MutableSequence[float]]:
        return [[self.logp.get(pretoken[n:n+(k+1)], -INFTY) for k in range(max_k)]
                for n in range(len(pretoken))]  # A matrix M where M[n][k] gives the log probability of the occurrence of the token starting at character n with size k+1.


class KudoPieceTokeniser(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, log_probabilities: Dict[str,float]):
        super().__init__(preprocessor, max_stepsize=max(map(len, log_probabilities)), objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ScoreGeneratorUsingSubstringClassifier(UnigramClassifier(log_probabilities)),
                score_combiner=ScoreSum()
            )
        ])
