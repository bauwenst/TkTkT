"""
Guided score functions. These are informed by knowledge of the language, e.g. models that estimate the probability of a
token appearing at all (with or without considering context), or models that estimate the probability of a split point.

TODO: Another idea:
    - There is still quite a big difference between a StringClassifier and the idea of turning a CharacterClassifier
      into a StringClassifier, which is that a StringClassifier normalises across all possible steps from length 1 to K
      that you can take (e.g. a softmax over the vocab, although you lose mass because only a few of those are actually
      allowed as a step at the current position) whilst if you use boundary probabilities, you're only going to be normalised for each of the 2^k boundary configurations of
      a fixed length k.
"""
from typing import List, MutableSequence
from abc import abstractmethod

import numpy as np
import torch
from math import exp

from bpe_knockout.project.config import morphologyGenerator, Pℛ𝒪𝒥ℰ𝒞𝒯

from .framework import ViterbiStepScoreGenerator, ViterbiStepScores, INFTY
from ...preparation.boundaries import BoundaryMarkerLocation
from ...preparation.splitters import WhitespaceAndMarkerPretokeniser
from ...util.printing import sgnprint


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


class ProbabilityTransform:

    def __init__(self, negate_as_complement: bool=False):
        self.negate_as_complement = negate_as_complement

    @abstractmethod
    def probabilityToScore(self, p: float) -> float:
        pass

    @abstractmethod
    def scoreToProbability(self, s: float) -> float:
        pass

    def scoreToScoreOfComplement(self, score_p: float) -> float:
        """
        Map the score S(p) to the score S(1-p).
        """
        return self.probabilityToScore(1 - self.scoreToProbability(score_p))

    def complementOfScore(self, s: float) -> float:
        return -s

    def complement(self, s: float):
        if self.negate_as_complement:
            return self.complementOfScore(s)
        else:
            return self.scoreToScoreOfComplement(s)

    def __repr__(self):
        return self.__class__.__name__


class IdentityPT(ProbabilityTransform):

    def probabilityToScore(self, p: float) -> float:
        return p

    def scoreToProbability(self, s: float) -> float:
        return s


class LogPT(ProbabilityTransform):
    """
    You may wonder if it makes sense from the POV of numerical stability to let a classifier generate log probabilities,
    transform them to probability with exp(), and then convert back to the original space with log(). Isn't it safer to
    use the original predictions, then?

    Well, if you test numpy's answer to np.log(np.exp(L)), you'll find that it only breaks down around L == -729, which
    represents a probability of 10^{-317}. At that point, some rounding error doesn't matter.

    The reason for using log probabilities could be to pile many predictions on top of each other, at which point the
    conversion would start to matter: you won't find a prediction of 10^{-300}, but you might find 20 predictions of 10^{-15}.
    The transform is only applied to individual predictions, and it is transformed values that are combined, so the transform
    never interacts with those accumulated values either.
    """
    def probabilityToScore(self, p: float) -> float:
        return np.log(p)

    def scoreToProbability(self, s: float) -> float:  # This leads to a complement that looks like ln(1-p) == ln(1 - e^(ln p)) != -ln p
        return np.exp(s)


class LinearPT(ProbabilityTransform):
    """
    In the log domain, you have disproportional punishment for wrong decisions.

    If you use a linear transform instead with symmetric reward -1 and +1, making the wrong decision can be balanced out
    by making the right decision in a comparable situation: e.g., take two points have 70% and 30% probability of being
    a boundary. You say both are boundaries, then the reward is ( 2*0.7-1 ) + ( 2*(0.3)-1 ) = 0.4 + (-0.4) = 0.
    """

    def __init__(self, a: int, b: int, negate_as_complement: bool=False):
        super().__init__(negate_as_complement)
        self.a = a
        self.b = b

    def probabilityToScore(self, p: float):
        return self.a + p*(self.b-self.a)

    def scoreToProbability(self, s: float) -> float:
        return (s - self.a)/(self.b - self.a)

    def __repr__(self):
        return self.__class__.__name__ + f"({sgnprint(self.a)},{sgnprint(self.b)})" + ("_NegComp")*self.negate_as_complement


class PiecewisePT(ProbabilityTransform):

    def __init__(self, a: int, b: int, negate_as_complement: bool=False):
        super().__init__(negate_as_complement)
        self.a = a
        self.b = b

    def probabilityToScore(self, p: float):
        return 2*(p-0.5)*self.b if p > 0.5 else 2*(0.5 - p)*self.a

    def scoreToProbability(self, s: float) -> float:
        return 0.5 + s/(2*self.b) if s > 0 else 0.5 - s/(2*self.a)

    def __repr__(self):
        return self.__class__.__name__ + f"({sgnprint(self.a)},{sgnprint(self.b)})" + ("_NegComp")*self.negate_as_complement


class ScoreGeneratorUsingCharacterClassifier(ViterbiStepScoreGenerator):
    
    def __init__(self, point_model: CharacterClassifier, **kwargs):
        self.logprob_classifier = point_model

    def getHardBoundaries(self, string: str) -> List[int]:
        # To reiterate how indexing works in the Viterbi framework:
        #   - The step score at [n,k] is the score you get when you are at the split position BEFORE character n and take a step of k+1 characters.
        #   - The boundary probability at n is the probability of there being a split position AFTER character n.

        # If the proposed segmentation is "w|or|d", you get a mask [1, 0, 1, 0].
        # We turn it into "|w|or|d|" with mask [1,   1, 0, 1,   1].
        # Position i now says whether there is a boundary BEFORE character i,
        # with an extra position at the end for a boundary behind the last character.
        boundary_after_asmask = [1*(exp(ln) > 0.5) for ln in self.logprob_classifier.getPointLogProbabilities(string)]
        boundary_before_asmask = [1] + boundary_after_asmask
        boundary_before_asmask[-1] = 1
        boundary_before = np.nonzero(boundary_before_asmask)[0]  # np.nonzero produces one array PER dimension. No errors are thrown if you forget the [0], but the zip() below will be empty!
        return boundary_before.tolist()


class ScoreGeneratorUsingCharacterClassifierAndTransform(ScoreGeneratorUsingCharacterClassifier):

    def __init__(self, point_model: CharacterClassifier, transform: ProbabilityTransform):
        super().__init__(point_model)
        self.T = transform

    def getBoundaryScores(self, string: str):
        boundary_scores = list(map(self.T.probabilityToScore, map(exp, self.logprob_classifier.getPointLogProbabilities(string))))  # one entry for each character
        boundary_scores[-1] = 0  # Score you get from walking to the end is 0. I.e.: it's a good idea, unless you can do better by splitting.
        return boundary_scores

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.T.__repr__() + ")"


class BoundaryScoresChosen(ScoreGeneratorUsingCharacterClassifierAndTransform):
    """
    For Viterbi paths that maximise the score on the character boundaries they hit.

    If using log probabilities, hitting low probabilities is disproportionately punished. This is not the case for
    symmetric linear transforms, like [0,1] -> [-1,+1]. Note that even though such scores might resemble probabilities
    a lot, they should STILL be summed, not multiplied (as if they're still logarithms) when accumulated.

    Now, point models give a probability of each character having a split point after it or not.
    So why not use probabilities?

    We want a Viterbi path to accumulate as much of the probability mass as possible (hit all the predicted
    boundaries), but disincentivise cheating by oversegmenting. Let's say you sum across the boundaries you hit.
    Oversegmenting would then cause you to collect all boundary probability, without being punished for collecting 0s.
    Hence, it is better to use a linear transform like -1 + P*(1 - (-1)) = 2P-1: you are rewarded for stepping on a
    boundary and you are punished for stepping on a non-boundary.
    The alternative is to use multiplication, or equivalently to use logarithms, but beware that this gives outsized
    punishment to stepping on a non-boundary (a single non-boundary can make 20 boundary hits evaporate by multiplying
    by a very small number). This is also easy to see when you take the ln, where you are punished by -100 if you
    get it wrong but getting it right is an invisible +0.
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_scores = self.getBoundaryScores(string)
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            K = min(max_k, N-n)  # Example: you are at n == 5 and there are N == 7 characters. This means you still need to include character 5 and 6 in the segmentation, using the probabilities at n+0 and n+1. Hence, there are K == N-n steps available.
            for k in range(K):
                scores.set(n,k, boundary_scores[n+k])  # Example: you are at n == 0. For k == 1, you take the second-smallest step, to 2. The probability of hitting a boundary by doing so is the probability of a boundary being after 1 == n+k.

        return scores


class BoundaryScoresAll(ScoreGeneratorUsingCharacterClassifierAndTransform):
    """
    Uses point boundaries, assumed to be independent, to compute segment probabilities.
    Should still be used with sum accumulation.

    Let's say you have suggested splits abc|de|f. The probability of the segment "abc" is computed now not as
        P(boundary after char 2 | abcdef)
    but instead as
        P(boundary after char 2 | abcdef)*P(no boundary after char 1 | abcdef)*P(no boundary after char 0 | abcdef)

    This is exactly what is calculated when you use log scores (because the sum of the logs is the log of the product of
    the probabilities, and you get numerical stability as a bonus), so mathematically speaking, this should be the most
    accurate type of score, since maximising this objective maximises the joint distribution across all possible split points.
    Of course, when constraints are involved, this is no longer true.

    There's a funky equivalence here: you could convert the CharacterClassifier into a StringClassifier first, and
    then use that StringClassifier with its usual grid generator.

    TODO: As it turns out, for symmetric scores, this is mathematically equivalent to only considering boundaries. Given the preferred score
          at position i, you can either choose to split there or not. You get the score at that position when you do.
            - If it's not a split and you do split, you receive a score, but it's negative.
            - If it's not a split and you don't split, you don't receive a score, which is +1 over the alternative decision.
            - If it's a split and you don't split, you don't receive a score.
            - If it's a split and you do split, you receive a score and it's positive, which is +1 over the alternative decision.
         In other words, when you DON'T SPLIT somewhere, the net reward of that decision doesn't have to be accounted for
         explicitly because it already was.
         The reason why this is different in the LogProbability class is because the ln(1-p) is not -ln(p) and hence if
         you add ln(1-p) as extra "negative reward", this is NOT equivalent to missing ln(p), unlike subtracting ln(p) would be.
         |
         I wonder what happens when you have asymmetric rewards. We want to incentivise precision (because you need smaller
         gains in precision to get bigger gains in F1), which means disencentivising (bad) splitting as much as possible.
         Right now, for the same word, it is equally scored to have
            - 2 good splits + 2 bad splits == 1 good + 1 bad.
            - 2 good splits + 1 bad splits == 1 good + 0 bad.
        In the second case, even though the net score is the same, the second one has better precision (and worse recall).
        It's not obvious that you want to lower recall, but in any case, either use a tiebreaker objective or punish bad
        splits more: with weight -2, you would get
            - +2 + -4 != +1 -2
            - +2 + -2 != +1 +0
        The obvious danger is of course that you won't go get a reachable split if it requires one bad split.
        |
        One more thing to prove: if you have this kind of negative reward, is it useful to count untaken splits, or is
        it again mathematically equivalent?
            - If it's not a split (carried score -2) and you do split, you receive -2.
            - If it's not a split (carried score -2) and you don't split, you receive +0 (boundary-only) or +2 (joint), so joint gives twice the benefit for being correct. Net +2 vs. net +4.
            - If it's a split (carried score +1) and you don't split, you receive +0 (boundary-only) or -1 (joint).
            - If it's a split (carried score +1) and you do split, you receive +1, so again joint just gives twice the benefit for being correct. Net +1 vs. net +2.
        There is also an alternative, keeping the positive and negative ranges fixed:
            - If it's not a split (carried score -2) and you do split, you receive -2.
            - If it's not a split (carried score -2) and you don't split, you receive +0 (boundary-only) or +1 (joint). Net +2 vs. net +3.
            - If it's a split (carried score +1) and you don't split, you receive +0 (boundary-only) or -2 (joint).
            - If it's a split (carried score +1) and you do split, you receive +1. Net +1 vs. net +3.
        In one, the net reward for being correct is doubled from boundary-only to joint.
        In the other, the net reward for a correct positive versus a correct negative is different (boundary-only) or the same (joint).
        |
        To figure out which of these two effects results in an equivalence between boundary-only and joint, you should
        implement both strategies and run all 4 variants. There are two ways to implement these asymmetric ranges:
            - Linear: min + P*(max-min)
            - Piecewise: (2*P-1)*max if P > 0.5 else (1-2*P)*min
        So actually, there are 8 variants that test 4 functions.
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_scores = self.getBoundaryScores(string)
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for n in range(N):
            K = min(max_k, N-n)  # Like above.
            for k in range(K):
                cumulative_score = boundary_scores[n+k]  # Like above.
                for i in range(k):
                    cumulative_score += self.T.complement(boundary_scores[n+i])
                scores.set(n, k, cumulative_score)

        return scores


###########################################################################################


class HardBoundaryPrefixLength(ScoreGeneratorUsingCharacterClassifier):
    """
    Takes the argmax segmentation as suggested by the classifier (which we know scores 92% in all metrics)
    and attributes a score of S to every step of length S that starts on a suggested segmentation boundary and ends on
    or before the next segmentation boundary.

    For example, for the suggested segmentation
        re|anim|atie|techn|iek
    the path scores for
        re|ani|mat|i|e|tec|hniek
    would be
        2 +3  +0  +0+0+3  +0
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaries(string)
        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)
        for start,end in zip(boundary_before[:-1], boundary_before[1:]):
            for k in range(min(end-start, max_k)):  # You should read this without the min(..., max_k) because realistically, boundaries will be pretty close together.
                scores.set(start, k, k+1)

        return scores


class HardBoundaryAndNonBoundaryPrefixLength(ScoreGeneratorUsingCharacterClassifier):
    """
    Same as the first one, except you get punished for making unnecessarily many steps when you do not start on a
    proposed boundary, to incentivise bridging the gap ASAP.

    What we HAVE is an incentive to not use short tokens when starting on a boundary.
    What we DON'T HAVE is an incentive to not use many tokens when not starting on a boundary.

    TODO: Although having a punishment of -1 per step is sufficient to incentivise choosing the path with least tokens
          to get to the next boundary, it may also help in being a counterweight to very large tokens being preferred
          and prioritised if it means ruining the rest of the segmentation.
          So, try increasing it to -2 and see what happens.
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaries(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=-1)  #   # The only difference with the previous class. Every step that is NOT a prefix is discouraged, and more such steps are more discouraged.
        for start,end in zip(boundary_before[:-1], boundary_before[1:]):
            for k in range(min(end-start, max_k)):
                scores.set(start, k, k+1)

        return scores


class HardBoundaryPrefixLengthExtended(ScoreGeneratorUsingCharacterClassifier):
    """
    The same as HardBoundaryPrefixLength except you are also rewarded if you start on a boundary and end AFTER the next
    boundary, with the reward stagnating at that boundary.

    In HardBoundaryPrefixLength, the reward drops to 0 after that boundary, which doesn't really allow making the
    trade-off of "I can capture this boundary if I jump over the next one". With boundary probabilities, you can jump
    a gap and still harvest the score for the boundary you end up at!
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaries(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=0)

        for n, n_where_reward_stagnates in zip(boundary_before[:-1], boundary_before[1:]):
            K = min(max_k, N-n)  # Same expression as for the probabilistic objectives, because despite the boundaries being indexed differently here, the index n in the outer loop has always been in grid coordinates (n == 0 meaning "when you are standing before the first character").
            for k in range(K):
                steps_with_increasing_reward = n_where_reward_stagnates - n
                if k < steps_with_increasing_reward:  # E.g.: if you have |wo|rd|, the top loop will produce (n,end) == (0,2), so taking a step of k+1 == 1 will get you reward k+1 == 1, a step of k+1 == 2 will get k+1 == 2, but a step of k+1 == 3 will still give reward end-n == 2. You hence got 2 iterations of increasing reward.
                    scores.set(n, k, k+1)
                else:
                    scores.set(n, k, steps_with_increasing_reward)

        return scores


class HardBoundaryAndNonBoundaryPrefixLengthExtended(ScoreGeneratorUsingCharacterClassifier):

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaries(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=-1)  # The only difference with the previous class

        for n, n_where_reward_stagnates in zip(boundary_before[:-1], boundary_before[1:]):
            K = min(max_k, N-n)
            for k in range(K):
                steps_with_increasing_reward = n_where_reward_stagnates - n
                if k < steps_with_increasing_reward:
                    scores.set(n, k, k+1)
                else:
                    scores.set(n, k, steps_with_increasing_reward)

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
        self.pretoken_shift = len(self.pretokeniser.marker.substitute)*(self.pretokeniser.marker.location == BoundaryMarkerLocation.START)
        self.gold_segmentations = {obj.lemma(): obj.morphSplit() for obj in morphologyGenerator()}

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        labels = np.zeros(len(pretoken), dtype=np.float32)

        word, _ = self.pretokeniser.marker.isolate(pretoken)
        if word in self.gold_segmentations:
            tokens = self.gold_segmentations[word].split()
            split_positions = np.cumsum([len(t) for t in tokens[:-1]]) - 1
            split_positions += self.pretoken_shift  # If "word" is shown to the character model as "Ġword", the suggested split indices should shift by 1.

            labels[split_positions] = 1

        return np.log(labels)


from transformers.models.canine.modeling_canine import CanineForTokenClassification, TokenClassifierOutput
from transformers.models.canine.tokenization_canine import CanineTokenizer
from ...util.environment import DEVICE


class HuggingFaceCharacterModelForTokenClassification(CharacterClassifier):
    """
    NOTE: Outputs log probabilities, not probabilities.
    """

    def __init__(self, characters_to_modelinput: CanineTokenizer, for_token_classification: CanineForTokenClassification,  # Sadly there is no generic "ForTokenClassification" type in HuggingFace's API, but any should work.
                 input_kwargs: dict=None):
        self.input_generator = characters_to_modelinput
        self.model           = for_token_classification
        self.generator_args = input_kwargs or dict()

        self.model.to(DEVICE)  # Speeds up inference about 2x to 4x on VSC. This call is in-place, unlike for tensors. https://stackoverflow.com/a/59560101/9352077

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        input_to_model = self.input_generator(pretoken, add_special_tokens=False, return_tensors="pt", **self.generator_args)
        with torch.no_grad():  # no_grad means all tensors returned don't have their gradient tracked, so you don't need to .detach() them before going to numpy.
            input_to_model = {k: v.to(DEVICE) for k,v in input_to_model.items()}
            prediction: TokenClassifierOutput = self.model(**input_to_model)

        chars_by_classes = prediction.logits.squeeze()  # Remove batch dimension (because it has size 1).
        normalisation_constants = torch.logsumexp(chars_by_classes, dim=1)  # Note that normalisation happens not OVER characters, but PER character. It happens over two binary classes, N times.
        positive_logits = chars_by_classes[:,1]
        logprobabilities = positive_logits - normalisation_constants
        return logprobabilities.cpu().numpy()  # Always need to go to CPU to cast down to numpy.
