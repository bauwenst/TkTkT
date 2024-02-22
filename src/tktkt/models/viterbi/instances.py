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

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=1,
                score_generator=ConstrainVocabulary(MaximiseTokenLength(), vocab, reset_value=0),
                score_combiner=Times()
            )
        ])


class HFModelViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int],
                 huggingface_checkpoint: str, tokeniser_class: Type[PreTrainedTokenizer], model_class: Type[PreTrainedModel]):
        max_step = max_step or max(len(t) for t in vocab)
        probability_model = HuggingFaceCharacterModelForTokenClassification(
            tokeniser_class.from_pretrained(huggingface_checkpoint),
            model_class.from_pretrained(huggingface_checkpoint)
        )

        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=ConstrainVocabulary(MaximiseSplitsOnBoundaries(probability_model), vocab, reset_value=-INFTY),
                score_combiner=Plus()
            )
        ])
