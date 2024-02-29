"""
Some examples of common Viterbi objectives.
"""
from typing import Type, Optional

from transformers import PreTrainedTokenizer, PreTrainedModel

from ...interfaces.tokeniser import Preprocessor, Vocab
from .framework import *
from .accumulators import *
from .objectives_unguided import *
from .objectives_guided import *
from .objectives_postprocessors import *


class LeastTokenViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(ConstantScore(), vocab, reset_value=+INFTY),
                score_combiner=ScoreSubtract()
            ),
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(TokenLength(), vocab, reset_value=-INFTY),
                score_combiner=ScoreMax()
            )
        ])


class ProductViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=1,
                score_generator=ConstrainVocabulary(TokenLength(), vocab, reset_value=0),
                score_combiner=ScoreProduct()
            )
        ])


class HFPointViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int],
                 cumulative_objective: bool, symmetric_scores: bool,
                 huggingface_checkpoint: str, tokeniser_class: Type[PreTrainedTokenizer], model_class: Type[PreTrainedModel], tokeniser_kwargs: dict=None):
        max_step = max_step or max(len(t) for t in vocab)

        # The thing that generates (log) probabilities
        probability_model = HuggingFaceCharacterModelForTokenClassification(
            tokeniser_class.from_pretrained(huggingface_checkpoint),
            model_class.from_pretrained(huggingface_checkpoint),
            tokeniser_kwargs
        )

        # The thing that uses probabilities to create Viterbi scores
        if not symmetric_scores:
            if cumulative_objective:
                generator_class = BoundaryAndNonBoundaryLogProbability
            else:
                generator_class = BoundaryLogProbability
        else:
            if cumulative_objective:
                generator_class = SymmetricBoundaryAndNonBoundaryProbability
            else:
                generator_class = SymmetricBoundaryProbability

        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(generator_class(probability_model), vocab, reset_value=-INFTY),
                score_combiner=ScoreSum()
            )
        ])
