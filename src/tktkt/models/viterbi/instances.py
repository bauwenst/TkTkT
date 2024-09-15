"""
Some examples of common Viterbi objectives.
"""
from typing import Type, Optional
from typing_extensions import Self

from ..viterbi import *
from ...interfaces.tokeniser import Preprocessor, Vocab


class LeastTokenViterbi(ViterbiTokeniser):
    """
    Minimises the amount of tokens in the result; the tiebreaker is that you maximise the length of the biggest token.
    """

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
    """
    Maximise the product of the lengths of all the tokens.
    Has weird prioritisation. For example, in a string of 6 characters, 1*1*4 < 1*5 < 6 == 2*3 < 2*2*2 == 2*4 < 3*3.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=1,
                score_generator=VocabularyConstraintExact(TokenLength(), vocab, reset_value=0),
                score_combiner=ScoreProduct()
            )
        ])


class BoMMa(ViterbiTokeniser):
    """
    The "Boundary Model Maximisation" tokeniser is a Viterbi tokeniser that uses a binary character classifier to
    generate probabilities at each inter-character position for whether there should be a split there, transforms those,
    and applies a vocabulary constraint on top to prevent the Viterbi optimiser from choosing some paths.

    You can make so many different models with this class that you can write an entire paper about just this one.
    """

    def __init__(self, preprocessor: Preprocessor, max_step: Optional[int],
                 score_generator: ScoreGeneratorUsingCharacterClassifier,
                 vocabulary_constraint_class: Type[VocabularyConstraint], vocab: Vocab):
        max_step = max_step or max(len(t) for t in vocab)

        self._score_generator = score_generator
        self._constraint = vocabulary_constraint_class(self._score_generator, vocab, reset_value=-INFTY)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                initial_score=self._getInitialScore(),
                score_generator=self._constraint,
                score_combiner=self._getAccumulator()
            )
        ])

    @classmethod
    @abstractmethod
    def _getInitialScore(cls) -> float:
        pass

    @classmethod
    @abstractmethod
    def _getAccumulator(cls) -> ViterbiAccumulator:
        pass

    def from_object(self, classifier: CharacterClassifier) -> Self:
        """
        Set the backend probability model to any CharacterClassifier object, no matter where it came from.
        """
        self._score_generator.setBackend(classifier)
        return self

    def from_pretrained_hf(self, huggingface_checkpoint: str, tokeniser_kwargs: dict=None) -> Self:  # model_class: Type[PreTrainedModel], tokeniser_class: Type[PreTrainedTokenizer],
        """
        Set the backend probability model to a HuggingFace checkpoint.
        """
        return self.from_object(HuggingFaceForBinaryCharacterClassification(characterclassifier_checkpoint=huggingface_checkpoint, input_kwargs=tokeniser_kwargs))

    def getName(self):
        return self.__class__.__name__ + "(" + self._score_generator.__repr__() + " + " + self._constraint.__class__.__name__ + ")"


class BoMMa_Sum(BoMMa):
    @classmethod
    def _getInitialScore(cls) -> float:
        return 0

    @classmethod
    def _getAccumulator(cls) -> ViterbiAccumulator:
        return ScoreSum()


class BoMMa_Product(BoMMa):
    @classmethod
    def _getInitialScore(cls) -> float:
        return 1

    @classmethod
    def _getAccumulator(cls) -> ViterbiAccumulator:
        return ScoreProduct()


class LeastTokenViterbiWithProbabilityTiebreaker(ViterbiTokeniser):
    """
    Minimises the amount of tokens, using not token length as tiebreaker, but instead accumulated boundary probabilities
    at the points you decided to split.

    TODO: Any results generated for this class should be deleted because there was a bug that made the tiebreaker incorrect.
    """

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
                score_generator=VocabularyConstraintExact(BoundaryScoresChosen(IdentityPT()), vocab, reset_value=-INFTY),
                score_combiner=ScoreSum()
            )
        ])
        self.objectives[1].score_generator.nested_generator.setBackend(logprob_classifier)


class ProbabilityViterbiWithLeastTokenTiebreaker(ViterbiTokeniser):
    """
    Maximises semi-hard* boundaries, with minimal tokens as tiebreaker, swapping the above objectives.

    The idea of why you want to swap the objectives: there are possibly many solutions with minimal tokens amounts, but
    that doesn't mean any of them are actually any good, whether you tiebreak for boundary probability or not. On the
    other hand, we know that there is at least 1 segmentation that hits all boundaries (character segmentation), so by
    minimising the amount of tokens, you will still get something that hits all boundaries whilst expanding the tokens
    as much as possible within those boundaries.

    In other words: you get something that competes with character segmentation by also having all desired boundaries,
    but with (hopefully) bigger tokens than characters.

    *We clip probabilities because if not, it's useless to have a tiebreaker: indeed, if you have possible boundaries
     [0.000001, 0.95, 0.0000001, 0.000001, 0.00001, 0.95, 0.000001], it is always technically better to grab all probabilities.
     We have 3 discretisation levels since sometimes your model really is 50/50 undecided.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int],
                 logprob_classifier: CharacterClassifier, discretisation_steps: int=3):
        max_step = max_step or max(map(len, vocab))
        super().__init__(preprocessor=preprocessor, max_stepsize=max_step, objectives=[
            ViterbiObjective(
                initial_score=0,
                score_generator=
                    VocabularyConstraintExact(
                        DiscretiseScores(
                            BoundaryScoresChosen(
                                IdentityPT()  # No transformation of the probabilities. We want them raw to discretise them.
                            ),
                            minimum_score=0, maximum_score=1, discretisation_levels=discretisation_steps
                        ),
                        subword_vocabulary=vocab, reset_value=-INFTY
                    ),
                score_combiner=ScoreSum()
            ),
            ViterbiObjective(
                initial_score=0,
                score_generator=VocabularyConstraintExact(ConstantScore(), vocab, reset_value=+INFTY),
                score_combiner=ScoreSubtract()
            )
        ])
        self.objectives[0].score_generator.nested_generator.nested_generator.setBackend(logprob_classifier)