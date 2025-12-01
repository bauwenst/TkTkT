"""
Guided score functions. These are informed by knowledge of the language, e.g. models that estimate the probability of a
token appearing at all (with or without considering context), or models that estimate the probability of a split point.

TODO: Another idea:
    - There is still quite a big difference between a SubstringClassifier and the idea of turning a CharacterClassifier
      into a SubstringClassifier, which is that a SubstringClassifier normalises across all possible steps from length 1 to K
      that you can take (e.g. a softmax over the vocab, although you lose mass because only a few of those are actually
      allowed as a step at the current position) whilst if you use boundary probabilities, you're only going to be normalised for each of the 2^k boundary configurations of
      a fixed length k.
"""
from typing import List, MutableSequence
from abc import abstractmethod

import numpy as np
import torch
from math import exp, sin, pi, asin
from math import log as ln

from .framework import ViterbiStepScoreGenerator, ViterbiStepScores, INFTY
from ....util.printing import sgnprint


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


class SubstringClassifier:

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


class MultiplicativelyBalancedProbabilityTransform(ProbabilityTransform):
    """
    Defines a subset of the probability transforms f(p) that satisfy
        1. f(p) is monotonously increasing.
        2. f(p) * f(1-p) = 1
        3. lim_{p->0.5+} f(p) = lim_{p->0.5-} f(p) = 1, that is, there is no discontinuity at p = 0.5.

    ---

    An example of a function that trivially satisfies the first two conditions yet doesn't satisfy the last is
        f(p) = 1/(1-p) for p > 0.5 and f(p) = p otherwise.
    (1) The separate pieces are monotonous, and for any p1 < 0.5 < p2, note that p2 < 0.5 => 0.5 < 1-p2 => 1/0.5 < 1/(1-p2) and thus f(p1) = p1 < 0.5 < 2 < 1/(1-p2) = f(p2).
    (2a) For any p' that is < 0.5, this gives f(p') * f(1-p') = p' * 1/(1 - (1-p')) = p' * 1/p' = 1.
    (2b) For any p' that is > 0.5, this gives f(p') * f(1-p') = 1/(1-p') * (1-p') = 1.
    (3) Yet lim_{p->0.5+} f(p) = 1/(1-0.5) = 1/0.5 = 2 whereas lim_{p->0.5-} f(p) = 0.5.

    ---

    This class achieves all three conditions by taking any auxiliary function g(x) for which
        1. g(0) = 1
        2. g(p) is monotonously increasing for p in [0,0.5]
    and then taking the piecewise function
        x > 0.5: f(x) = g(x - 0.5)
        x < 0.5: f(x) = 1/g(0.5 - p)
    """

    def probabilityToScore(self, p: float) -> float:
        return 1/self.isOneAtZero(0.5 - p) if p < 0.5 else self.isOneAtZero(p - 0.5)

    def scoreToProbability(self, s: float) -> float:
        return 0.5 - self.isOneAtZero_inv(1/s) if s < 1 else 0.5 + self.isOneAtZero_inv(s)

    @abstractmethod
    def isOneAtZero(self, x: float) -> float:
        pass

    @abstractmethod
    def isOneAtZero_inv(self, y: float) -> float:
        pass


class PowerMBPT(MultiplicativelyBalancedProbabilityTransform):
    """
    Models g(x) = 1 + c*x^p.
    Also, the case p = 1 has almost the same resulting f(x) as g(x) = 1 + c*ln(1+x) and g(x) = 1 + c*tan(x), hence we
    don't have a subclass for those.
    """

    def __init__(self, power: float, scale: float, negate_as_complement: bool=False):
        super().__init__(negate_as_complement=negate_as_complement)
        assert power != 0 and scale != 0
        self.scale = scale
        self.power = power

    def isOneAtZero(self, x: float) -> float:
        return 1 + self.scale*pow(x, self.power)

    def isOneAtZero_inv(self, y: float) -> float:
        return pow((y-1)/self.scale, 1/self.power)

    def __repr__(self):
        return self.__class__.__name__ + f"(c={self.scale},p={sgnprint(self.power)})"


class DoublingMBPT(PowerMBPT):
    """
    Models g(x) = 1 + 2*x, which is one specific case of PowerMBPT that is easy to compute.
    It turns P = 100% into a score factor ×2 and turns P = 0% into a score factor ×1/2.
    """

    def __init__(self):
        super().__init__(power=1, scale=2)

    def isOneAtZero(self, x: float) -> float:
        return 1 + 2*x

    def isOneAtZero_inv(self, y: float) -> float:
        return (y-1)/2

    def __repr__(self):
        return f"PowerMBPT(c={self.scale},p={sgnprint(self.power)})"


class ExponentialMBPT(MultiplicativelyBalancedProbabilityTransform):
    """
    Models g(x) = exp(c*x).
    """

    def __init__(self, scale: float, negate_as_complement: bool=False):
        super().__init__(negate_as_complement=negate_as_complement)
        assert scale != 0
        self.scale = scale

    def isOneAtZero(self, x: float) -> float:
        return exp(self.scale*x)

    def isOneAtZero_inv(self, y: float) -> float:
        return ln(y)/self.scale

    def __repr__(self):
        return self.__class__.__name__ + f"(c={self.scale})"


class SineMBPT(MultiplicativelyBalancedProbabilityTransform):
    """
    Models g(x) = 1 + c/4 * (1 + sin(2*pi*(x - 1/4))) which has the same extrema as PowerMBPT of the same c and p = 1,
    but a different flow.
    """

    def __init__(self, scale: float, negate_as_complement: bool=False):
        super().__init__(negate_as_complement=negate_as_complement)
        assert scale != 0
        self.scale = scale

    def isOneAtZero(self, x: float) -> float:
        return 1 + 0.25*self.scale*(1 + sin(2*pi*(x - 0.25)))

    def isOneAtZero_inv(self, y: float) -> float:
        return 1/(2*pi) * asin((y-1)*4/self.scale - 1) + 0.25

    def __repr__(self):
        return self.__class__.__name__ + f"(c={self.scale})"


########################################################################################################################


class ScoreGeneratorUsingCharacterClassifier(ViterbiStepScoreGenerator):
    """
    Stores a model that generates log(probability) values.
    The reason why that model is not a constructor argument is because the constructor of these score generators should
    be used to declare how the generator works ON TOP OF the model REGARDLESS OF which model it is. The backend model
    can vary across runtimes, whilst the parameters of the generator are fixed. "I want this tokeniser to use a (-2,+1)
    linear probability transform" is much more static than "I want this tokeniser to get its probabilities from this checkpoint".

    You will receive a NoneTypeException at some point if you forgot to set the backend.

    This class has two inheritors: one transforms the probabilities of the model and puts the result directly in the
    score grid, the other thresholds the probabilities and uses the location of these hard boundaries to generate its own scores.
    """
    def __init__(self):
        self.logprob_classifier: CharacterClassifier = None

    def setBackend(self, point_model: CharacterClassifier):
        self.logprob_classifier = point_model


class ScoreGeneratorUsingCharacterClassifierForTransform(ScoreGeneratorUsingCharacterClassifier):
    """
    Stores a transformation for probabilities (not log probabilities!), since it automatically converts the given model's
    log probabilities into probabilities before applying the transformation.
    """

    def __init__(self, transform: ProbabilityTransform):
        super().__init__()
        self.T = transform

    def getBoundaryScores(self, string: str) -> List[float]:
        boundary_scores = list(map(self.T.probabilityToScore, map(exp, self.logprob_classifier.getPointLogProbabilities(string))))  # one entry for each character
        boundary_scores[-1] = self.T.probabilityToScore(0.5)  # Score you get from walking straight to the end is indifferent. I.e.: doing it is not punished so much you take other bad splits, but it's not rewarded so much you don't go looking for better splits.
        return boundary_scores

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.T.__repr__() + ")"


class BoundaryScoresChosen(ScoreGeneratorUsingCharacterClassifierForTransform):
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

    Furthermore, multiplying probabilities (summing logarithms) has some bias towards shorter token sequences.
    Picking up [START, 1, 1, 1, 0.99, END] is worse than picking up [START, 1, 1, 1, END] when you multiply, even though
    clearly the first is much better. In log space, you get -0 -0 -0 -0.000001 versus -0 -0 -0. This is why a probability
    transform and then summing instead of multiplying is worthwhile.
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


class BoundaryScoresAll(ScoreGeneratorUsingCharacterClassifierForTransform):
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

    There's a funky equivalence here: you could convert the CharacterClassifier into a SubstringClassifier first, and
    then use that SubstringClassifier with its usual grid generator.

    TODO: As it turns out, for symmetric scores (not regular probabilities), this is mathematically equivalent to only considering chosen boundaries.
          Given the preferred score at position i, you can either choose to split there or not. You get the score at that position when you do.
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


########################################################################################################################


class ScoreGeneratorUsingCharacterClassifierForHardBoundaries(ScoreGeneratorUsingCharacterClassifier):
    """
    Base class for all prefix/suffix score generators, which don't put predicted probabilities into the grid but rather
    some score related to the hard boundaries they indicate.
    """

    def __init__(self, punishment: float=-1, do_normalise: bool=False):
        """
        :param punishment: Default score of a step when it isn't involved with any boundary.
        :param do_normalise: Make the scores in the generated grids not exceed 1.0. For example, in prefix/suffix objectives,
                             the scores are as high as the length of the segments suggested by the hard boundaries. This means
                             more reward is given when you prioritise long tokens. Yet, arguably, you want to respect boundaries
                             for small morphemes like affices, so by normalising, you get more reward for getting 2 short affices right
                             than getting 1 long stem right.
        """
        super().__init__()
        self.punishment = -abs(punishment)  # User can give positive or negative values, doesn't matter.
        self.do_normalise = do_normalise

    def getHardBoundaryIndices(self, string: str) -> List[int]:
        """
        Returns the INDICES of which characters are preceded by a boundary with probability > 0.5.
        Always includes index 0 and len(string), which are imaginary boundaries.
        """
        # To reiterate how indexing works in the Viterbi framework:
        #   - The step score at [n,k] is the score you get when you are at the split position BEFORE character n (with n being 0-based) and take a step of k+1 characters.
        #   - The boundary probability at n is the probability of there being a split AFTER character n.
        # In other words: steps scores are indexed by inter-character positions (including two at the ends), boundary probabilities are indexed by character.

        # If the proposed segmentation is "w|or|d", the raw probability mask would be [1, 0, 1, 0].
        # Index n says whether there is a boundary AFTER character n.
        boundary_after_asmask = [1 * (exp(ln) > 0.5) for ln in self.logprob_classifier.getPointLogProbabilities(string)]
        # We turn it into "|w|or|d|" with mask [1,   1, 0, 1,   1].
        # Index n now says whether there is a boundary BEFORE character n,
        # with an extra position at the end for a boundary behind the last character.
        boundary_before_asmask = [1] + boundary_after_asmask
        boundary_before_asmask[-1] = 1
        # Finally, convert this to indices. In the above example: [0, 1, 3, 4].
        boundary_before = np.nonzero(boundary_before_asmask)[0]  # np.nonzero produces one array PER dimension. No errors are thrown if you forget the [0], but the zip() below will be empty!
        return boundary_before.tolist()

    def __repr__(self):
        return self.__class__.__name__ + "_Normed"*self.do_normalise + f"(pm={self.punishment}))"


class BoundaryPrefixLength(ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Doesn't output probability/score, but does character counting instead.

    Takes the argmax segmentation as suggested by the classifier (which we know scores 92% in all metrics)
    and attributes a score of S to every step of length S that starts on a suggested segmentation boundary and ends on
    or before the next segmentation boundary.

    For example, for the suggested segmentation
        re|anim|atie|techn|iek
    the path scores for
        re|ani|mat|i|e|tec|hniek
    would be
        2 +3  +0  +0+0+3  +0

    If the punishment is nonzero, you have an incentive against making unnecessarily many steps when you do not start on
    a proposed boundary, to try bridging the gap ASAP. (You already have incentive to use long tokens when you start on
    a boundary.)
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaryIndices(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=self.punishment)  # Every step that is NOT a prefix is discouraged, and more such steps are more discouraged.
        for start,end in zip(boundary_before[:-1], boundary_before[1:]):
            span_length = end-start
            denominator = span_length if self.do_normalise else 1

            K = min(span_length, max_k)
            for k in range(K):
                scores.set(start, k, (k+1)/denominator)  # The reason for using span_length rather than K as denominator is that we want the normalised vs. unnormalised behaviour to be the same a.f.o. changing K. Let's say you used K here, the max score would always be K/K == 1 no matter what K was, whereas the max unnormalised score would change with varying K (if K < span_length). Now, they both have a varying max with varying K.

        return scores


class BoundaryPrefixLengthExtended(ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    The same as BoundaryPrefixLength except you are also rewarded if you start on a boundary and end AFTER the next
    boundary, with the reward stagnating at that boundary.

    In BoundaryPrefixLength, the reward drops to 0 after that boundary (or the punishment), which doesn't really allow
    making the trade-off of "I can capture this boundary if I jump over the next one". With boundary probabilities, you
    can jump a gap and still harvest the score for the boundary you end up at!
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaryIndices(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=self.punishment)

        for n, n_where_reward_stagnates in zip(boundary_before[:-1], boundary_before[1:]):
            steps_with_increasing_reward = n_where_reward_stagnates - n  # == span_length above.
            denominator = steps_with_increasing_reward if self.do_normalise else 1

            K = min(max_k, N-n)
            for k in range(K):
                scores.set(n, k, min(steps_with_increasing_reward, k+1)/denominator)  # Increasing as long as k <= steps_with_increasing_reward-1, then stays at that value. E.g.: if you have |wo|rd|, the outer loop will produce (n,end) == (0,2), so taking a step of k+1 == 1 will get you reward k+1 == 1, a step of k+1 == 2 will get k+1 == 2, but a step of k+1 == 3 will still give reward end-n == 2. You hence got 2 iterations of increasing reward.

        return scores


class BoundarySuffixLength(ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Equivalent of BoundaryPrefixLength but for suffices: you get a score proportional to how much of the END of a
    suggested segment you capture.

    If you have splits |abcde|fghi| then you get a score of 3 for a token "cde" because it ENDS on a boundary, not
    starts on it. Prefix objectives give that 0 score, and rather give you credit for any extension of that split point
    (e.g. "fgh").
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaryIndices(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=self.punishment)

        for start,end in zip(boundary_before[:-1], boundary_before[1:]):
            span_length = end - start
            denominator = span_length if self.do_normalise else 1

            K = min(max_k, span_length)
            for k in range(K):
                scores.set(end-(k+1), k, (k+1)/denominator)

        return scores


class BoundarySuffixLengthExtended(ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Analogue of BoundaryPrefixLengthExtended.
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaryIndices(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=self.punishment)

        for n_where_reward_stagnates, n in zip(boundary_before[:-1], boundary_before[1:]):
            steps_with_increasing_reward = n - n_where_reward_stagnates
            denominator = steps_with_increasing_reward if self.do_normalise else 1

            K = min(max_k, n)  # K is the amount of steps you move back. If you are on index n, you move back 1 steps to n-1, 2 steps to n-2, ..., n steps to 0, and you can't go further.
            for k in range(K):
                scores.set(n-(k+1), k, min(steps_with_increasing_reward, k+1)/denominator)

        return scores


class BoundaryPrefixAndSuffixLengthExtended(ScoreGeneratorUsingCharacterClassifierForHardBoundaries):
    """
    Combination of prefix and suffix scores.
        - When you don't start on a boundary and don't end on a boundary, you get punished.
        - When you do start on a boundary but don't end on a boundary, you get prefix score.
        - When you don't start on a boundary and do end on a boundary, you get suffix score.
        - When you do start on a boundary and do end on a boundary, there are two cases:
            - There is at least one boundary in between them: you get the sum of prefix and suffix.
            - There are no boundaries in between them: you get prefix or suffix (they are equal).
              The reason for doing this is so that every character is counted exactly once.

    You might say: in a string |ab|cdefg|, doesn't the token "bcdefg" get you the same score as "cdefg", since the
    suffix score for hitting the "g" is still the 5 you'd get for "cdefg"? Yes, HOWEVER, notice that "bcdefg" starts on
    "b" rather than "c", so you didn't get a suffix score for reaching "c".
    Thus, /a/bcdefg/ contributes 1 ("a" contains 1 starting character of "ab") + 5 ("bcdefg" contains 5 ending characters of "cdefg") = 6,
    whilst /ab/cdefg/ contributes 2 ("ab" match) + 5 ("cdefg" match) = 7.

    A slight correction is added: the decimal part of the score stores how many exact token matches have been made. This
    is to break the following two ties: we know the best segmentation gets you 7. However,
        - The segmentation /ab/cde/fg/ gets you 2 ("ab" match) + 3 ("cde" prefix of "cdefg") + 2 ("fg" suffix of "cdefg") = 7.
        - The segmentation /abcdefg/ gets you 2 ("abcdefg" contains 2 starting characters of "ab") + 5 ("abcdefg" contains 5 ending characters of "cdefg") = 7.
    In other words: you can split a good token in half or merge two tokens without seeing the difference. To counter this,
    an exact match gets an extra 0.01 as natural tiebreaker, hence the scores become
        - /ab/cdefg/ = 2.01 + 5.01 = 7.02
        - /ab/cde/fg/ = 2.01 + 3 + 2 = 7.01
        - /abcdefg/ = 2 + 5 = 7
    """

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        boundary_before = self.getHardBoundaryIndices(string)

        N = len(string)
        scores = ViterbiStepScores(N, max_k, default=self.punishment)

        visited_indices = set()

        for start, end in zip(boundary_before[:-1], boundary_before[1:]):
            span_length = end-start
            denominator = span_length if self.do_normalise else 1

            # Prefix score
            K_nodes_forward = min(max_k, N-start)
            for k in range(K_nodes_forward):
                if (start,k) not in visited_indices:  # Reset the punishment to 0 so you can add to it.
                    visited_indices.add((start,k))
                    scores.set(start, k, 0)
                scores.add(start, k, min(span_length, k+1)/denominator)

            # Suffix score
            K_nodes_back = min(max_k, end)
            for k in range(K_nodes_back):
                if (end-(k+1), k) not in visited_indices:  # Reset the punishment to 0 so you can add to it.
                    visited_indices.add((end-(k+1), k))
                    scores.set(end-(k+1), k, 0)
                if k != span_length-1:  # Protection against double-counting this span's prefix and suffix.
                    scores.add(end-(k+1), k, min(span_length, k+1)/denominator)
                else:
                    scores.add(end-(k+1), k, 0.01/denominator)

        return scores


########################################################################################################################

from ....preparation.boundaries import BoundaryMarkerLocation, BoundaryMarker

class GoldSplits(CharacterClassifier):
    """
    Uses gold segmentations as split point suggestions. This is cheating, but it is a good baseline!

    FIXME: Has to be able to handle byte-based input too. What you receive for scoring is a pretoken about to be
           tokenised. That means for ë you will receive byte-based input and won't find it in your gold dictionary.
           preprocessor.undo() probably works.
    """

    def __init__(self, boundary_marker: BoundaryMarker):
        from ....preparation.splitters import OnWhitespaceAndAddMarker
        self.pretokeniser = OnWhitespaceAndAddMarker(replacement=boundary_marker)
        self.pretoken_shift = len(self.pretokeniser.marker.substitute)*(self.pretokeniser.marker.location == BoundaryMarkerLocation.START)

        from bpe_knockout.project.config import morphologyGenerator
        self.gold_segmentations = {obj.word: obj.segment() for obj in morphologyGenerator()}

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        labels = np.zeros(len(pretoken), dtype=np.float32)

        word, _ = self.pretokeniser.marker.isolate(pretoken)
        if word in self.gold_segmentations:
            tokens = self.gold_segmentations[word]
            split_positions = np.cumsum([len(t) for t in tokens[:-1]]) - 1
            # FIXME: Probably shouldn't add the shift to EVERY pretoken, e.g. "energie-efficiëntie" becomes ["Ġenergie", "-", "efficiëntie"] and that last one needs no shift.
            split_positions += self.pretoken_shift  # If "word" is shown to the character model as "Ġword", the suggested split indices should shift by 1.

            labels[split_positions] = 1

        return np.log(labels)


from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.models.canine.modeling_canine import CanineForTokenClassification, TokenClassifierOutput
from transformers.models.canine.tokenization_canine import CanineTokenizer
from ....util.environment import DEVICE


class HuggingFaceForBinaryCharacterClassification(CharacterClassifier):
    """
    Classifies characters using a HuggingFace checkpoint.
    """

    def __init__(self, characterclassifier_checkpoint: str, input_kwargs: dict=None):
        self.characters_to_modelinput: CanineTokenizer = AutoTokenizer.from_pretrained(characterclassifier_checkpoint)
        self.input_kwargs = input_kwargs or dict()

        # Sadly there is no generic "ForTokenClassification" type in HuggingFace's API nor is there any way to check that
        # the model is actually classifying tokens, so there's no real way to enforce statically that the user actually gives the correct checkpoint.
        self.model_for_tokenclassification: CanineForTokenClassification = AutoModelForTokenClassification.from_pretrained(characterclassifier_checkpoint)
        self.model_for_tokenclassification.to(DEVICE)  # Speeds up inference about 2x to 4x on VSC. This call is in-place, unlike for tensors. https://stackoverflow.com/a/59560101/9352077

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        model_input = self.characters_to_modelinput(pretoken, add_special_tokens=False, return_tensors="pt", **self.input_kwargs)
        with torch.no_grad():  # no_grad means all tensors returned don't have their gradient tracked, so you don't need to .detach() them before going to numpy.
            model_input = {k: v.to(DEVICE) for k,v in model_input.items()}
            prediction: TokenClassifierOutput = self.model_for_tokenclassification(**model_input)

        chars_by_classes = prediction.logits.squeeze()  # Remove batch dimension (because it has size 1).
        normalisation_constants = torch.logsumexp(chars_by_classes, dim=1)  # Note that normalisation happens not OVER characters, but PER character. It happens over two binary classes, N times.
        positive_logits = chars_by_classes[:,1]
        logprobabilities = positive_logits - normalisation_constants   # TODO: Can this not just be done with torch.nn.LogSoftmax? A softmax's denominator would be SumExp, so subtracting LogSumExp is the same as taking Log of softmax division, hence LogSoftmax.
        return logprobabilities.cpu().numpy()[:len(pretoken)]  # Always need to go to CPU before casting down to numpy. The slice is needed because models like CANINE add padding characters up to a given amount.


########################################################################################################################


class ScoreGeneratorUsingSubstringClassifier(ViterbiStepScoreGenerator):
    """
    The Viterbi score grid is 2D. A substring classifier is also 2D. Therefore, there is no extra step required to
    interpret the probabilities of the classifier into a 2D grid, unlike for character models.

    Has belated instantiation of the model for the same reason as the character classifiers.
    """

    def __init__(self):
        self.classifier: SubstringClassifier = None

    def setBackend(self, classifier: SubstringClassifier):
        self.classifier = classifier

    def generateGrid(self, string: str, max_k: int) -> ViterbiStepScores:
        scores = ViterbiStepScores(len(string), max_k)
        scores.grid = np.array(self.classifier.getSegmentLogProbabilities(string, max_k))
        return scores
