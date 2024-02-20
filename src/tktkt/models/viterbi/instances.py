"""
Some examples of common Viterbi objectives.
"""
from typing import Type

from transformers import PreTrainedTokenizer, PreTrainedModel

from ...interfaces.general import Pretokeniser, Vocab
from .framework import *
from .accumulators import *
from .objectives_unguided import *
from .objectives_guided import *
from .objectives_postprocessors import *


class LeastTokenViterbi(ViterbiTokeniser):

    def __init__(self, pretokeniser: Pretokeniser, max_step: int, vocab: Vocab):
        super().__init__(pretokeniser, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(MinimiseTokenAmount(), vocab, reset_value=-INFTY),
                score_combiner=Plus()
            ),
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(MaximiseTokenLength(), vocab, reset_value=-INFTY),
                score_combiner=Max()
            )
        ])


class ProductViterbi(ViterbiTokeniser):

    def __init__(self, pretokeniser: Pretokeniser, max_step: int, vocab: Vocab):
        super().__init__(pretokeniser, max_step, objectives=[
            ViterbiObjective(
                initial_score=1,
                score_generator=ConstrainVocabulary(MaximiseTokenLength(), vocab, reset_value=0),
                score_combiner=Times()
            )
        ])


class HFModelViterbi(ViterbiTokeniser):

    def __init__(self, pretokeniser: Pretokeniser, max_step: int, vocab: Vocab,
                 huggingface_checkpoint: str, tokeniser_class: Type[PreTrainedTokenizer], model_class: Type[PreTrainedModel]):
        probability_model = HuggingFaceCharacterModelForTokenClassification(
            tokeniser_class.from_pretrained(huggingface_checkpoint),
            model_class.from_pretrained(huggingface_checkpoint)
        )

        super().__init__(pretokeniser, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(MaximiseSplitsOnBoundaries(probability_model), vocab, reset_value=-INFTY),
                score_combiner=Plus()
            )
        ])
