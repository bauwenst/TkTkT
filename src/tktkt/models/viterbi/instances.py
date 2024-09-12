"""
Some examples of common Viterbi objectives.
"""
from typing import Type, Optional

from transformers import PreTrainedTokenizer, PreTrainedModel

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


class HFPointViterbi(ViterbiTokeniser):
    """
    Uses a HuggingFace character model for generating the probability that a split point should occur after each
    character, transforms those probabilities into Viterbi scores, and applies a vocabulary constraint to prevent
    the Viterbi optimiser from choosing some paths.

    You can make so many different models with this class that you can write an entire paper about just this one.
    """

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
                score_generator=VocabularyConstraintExact(BoundaryScoresChosen(logprob_classifier, IdentityPT()), vocab, reset_value=-INFTY),
                score_combiner=ScoreSum()
            )
        ])


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
                                logprob_classifier, IdentityPT()  # No transformation of the probabilities. We want them raw to discretise them.
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


class MultiplicativeBalanceViterbi(ViterbiTokeniser):
    """
    The idea is this: if you have a boundary model that gives you probabilities at each character boundary, one thing
    you might do to score a segmentation is multiply the probabilities of the boundaries you choose to split on.
    The problem here is that it disincentivises gathering a higher amount of high-probability boundaries even if it takes
    no other bad splits to do so: 0.9999 * 0.9999 * 0.9999 is smaller than 0.9999 * 0.9999. A second issue is that when
    you do hit a low-probability boundary, you carry that forever: 0.1 * 0.9999 * 0.9999 * 0.9999 * 0.9999 * 0.9999
    has 5 perfect splits and 1 bad split, and it is 10x smaller than having even 1 good split at 0.9999.

    So, what you want is to give all probabilities past 50% a boost so that you can recover completely from a bad split
    and so that two perfect splits are better than one perfect split.
    """
    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, max_step: Optional[int],
                 logprob_classifier: CharacterClassifier, transform: MultiplicativelyBalancedProbabilityTransform):
        max_step = max_step or max(map(len, vocab))
        super().__init__(preprocessor=preprocessor, max_stepsize=max_step, objectives=[
            ViterbiObjective(
                initial_score=1,
                score_generator=
                    VocabularyConstraintExact(
                        BoundaryScoresChosen(
                            logprob_classifier, transform
                        ),
                        subword_vocabulary=vocab, reset_value=-INFTY
                    ),
                score_combiner=ScoreProduct()
            )
        ])

    def getName(self):
        constraint: VocabularyConstraint = self.objectives[0].score_generator
        return self.__class__.__name__ + "(" + constraint.nested_generator.__repr__() + " + " + constraint.__class__.__name__ + ")"
