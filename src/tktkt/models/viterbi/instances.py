"""
Some examples of common Viterbi objectives.
"""
from typing import Type, Optional

from transformers import PreTrainedTokenizer, PreTrainedModel

from ...interfaces.tokeniser import Preprocessor, Vocab
from ..viterbi import *


class LeastTokenViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=VocabularyConstraintExact(ConstantScore(), vocab, reset_value=+INFTY),
                score_combiner=ScoreSubtract()
            ),
            ViterbiObjective(
                initial_score=0,
                score_generator=VocabularyConstraintExact(TokenLength(), vocab, reset_value=-INFTY),
                score_combiner=ScoreMax()
            )
        ])


class ProductViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=1,
                score_generator=VocabularyConstraintExact(TokenLength(), vocab, reset_value=0),
                score_combiner=ScoreProduct()
            )
        ])


class HFPointViterbi(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int],
                 # Grid generator
                 score_generator_class: Type[ScoreGeneratorUsingCharacterClassifier], score_transform: Optional[ProbabilityTransform], vocabulary_constraint_class: Type[VocabularyConstraint],
                 # Probability generator
                 huggingface_checkpoint: str, tokeniser_class: Type[PreTrainedTokenizer], model_class: Type[PreTrainedModel], tokeniser_kwargs: dict=None):
        max_step = max_step or max(len(t) for t in vocab)

        # The thing that generates (log) probabilities
        probability_model = HuggingFaceCharacterModelForTokenClassification(
            tokeniser_class.from_pretrained(huggingface_checkpoint),
            model_class.from_pretrained(huggingface_checkpoint),
            tokeniser_kwargs
        )

        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=vocabulary_constraint_class(score_generator_class(probability_model, transform=score_transform), vocab, reset_value=-INFTY),
                score_combiner=ScoreSum()
            )
        ])

    def getName(self):
        constraint: VocabularyConstraint = self.objectives[0].score_generator
        return self.__class__.__name__ + "(" + constraint.nested_generator.__repr__() + " + " + constraint.__class__.__name__ + ")"


class LeastTokenWithHfTiebreaker(ViterbiTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int],
                 logprob_classifier: CharacterClassifier):
        max_step = max_step or max(map(len, vocab))
        super().__init__(preprocessor=preprocessor, max_stepsize=max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=VocabularyConstraintExact(ConstantScore(), vocab, reset_value=+INFTY),
                score_combiner=ScoreSubtract()
            ),
            ViterbiObjective(
                initial_score=0,
                score_generator=VocabularyConstraintExact(BoundaryScoresChosen(logprob_classifier, IdentityPT()), vocab, reset_value=+INFTY),
                score_combiner=ScoreSum()
            )
        ])
