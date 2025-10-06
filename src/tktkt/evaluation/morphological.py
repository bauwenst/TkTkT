"""
Taken from the BPE knockout repo.
"""
from typing import Dict, List
from dataclasses import dataclass

import json

from modest.interfaces.morphologies import MorphologyVisitor, MorphSplit, FreeMorphSplit
from modest.interfaces.datasets import ModestDataset, M

from ..util.aggregates import ConfusionMatrix, NestedAverage, NestedMicroMacro
from ..util.iterables import cumsum
from ..paths import TkTkTPaths
from ..util.types import HoldoutState
from .observing import *


def compareSplits_cursors(candidate: str, reference: str):
    """
    Takes two words split with spaces and computes the factors of the precision and recall of those splits.
    For example,
        candidate a bc d ef
        reference a b cd e f
    has precision 66% and recall 75%.

    Assumes they have the same amount of non-spaces.
    """

    candidate_index = 0
    reference_index = 0

    tp        = 0
    relevant  = 0
    predicted = 0
    total     = 0
    while candidate_index < len(candidate) and reference_index < len(reference):
        candidate_split = candidate[candidate_index] == " "
        reference_split = reference[reference_index] == " "

        tp += candidate_split and reference_split
        relevant  += reference_split
        predicted += candidate_split
        total += 1

        candidate_index += 1 + candidate_split
        reference_index += 1 + reference_split

    return tp, predicted, relevant, total - 1  # The `total` variable counts the amount of characters, not splits.


def morphFraction(morphs: List[str], tokens: List[str], output: NestedMicroMacro):
    """
    For each morph, find the token which contains most of its characters and compute the fraction of its total length
    that is present in that token. Then micro- or macro-average this across morphs in the corpus.

    Given is the reference AAA BBB CCC. When treating morphological alignment as binary classification,
    a candidate AAA BBBC CC shows up the same as AAA BBBCC C (both have two splits, of which one is precise
    and both recall one of two splits). Meanwhile, we want to have a metric that measures the fact that
    the second one has a token dedicated to the last morpheme which will be much less informative because most of the
    morpheme has disappeared.

    More examples that illustrate this concept:

        Reference:
        AAA BBB CCC

        Candidates:
        AA ABBB CCC    punish for BBB being distorted +1 and AAA being distorted -1 (their majorities are still part of one token), or equivalently, the distance of 1 moved by the split.
        AA ABBBC CC    idem except two splits with distance 1
        A AABBBCC C    trickier; the majority of the three morphemes are in one token. you could say two boundaries with distance 2, but arguably, there not being a separate majority-A token means that its meaning is no longer represented
        AAAB BBC CC    both boundaries shifted; equivalent to the second case.
        AAABB BCC C    BCC is majority-C, AAABB is majority-A, so no B-specific token. the last split is oversegmentation rather than an offset split between B and C. Indeed, there is a distance limit to how far you can align a split with a reference.
        AAABBB CC C    oversegmentation of CCC.
        AAABBB CCC     lack of A/B morpheme.
        AAABBBC CC
        AAABBBCC C

    This function has the following trade-off:
        - Pro: does not have to align splits, so also has no issue when there are more/less splits in reference vs. candidate.
        - Con: it's somewhat of a precision metric because NOT splitting at all gives you a score of 100% (yet zero recall).
    """
    morph_splits = list(cumsum(map(len, morphs)))
    token_splits = list(cumsum(map(len, tokens)))

    best_token_length = 0
    token_start = 0

    morph_split_index = 0
    morph_start = 0
    morph_end = morph_splits[morph_split_index]

    for index in token_splits:
        while index >= morph_end:  # In this case, the morph is complete.
            # Finish morph
            max_length = morph_end - morph_start
            token_length = morph_end - token_start
            best_token_length = max(best_token_length, token_length)

            output.add(best_token_length, max_length)

            morph_split_index += 1
            if morph_split_index < len(morph_splits):
                # Start new morph
                morph_start = morph_end
                morph_end = morph_splits[morph_split_index]
            else:
                morph_end = float("inf")

            # What remains of the token will be used as if the token was always only that remainder.
            token_start = morph_start
            best_token_length = 0

        token_length = index - token_start
        best_token_length = max(best_token_length, token_length)
        token_start = index

    output.fence()


def alignedSegmentationScore(morphs: List[str], tokens: List[str], adversarial: bool, output: NestedAverage):
    """
    For each reference split, compute the distance to the nearest candidate split. Normalise these distances somehow,
    then again micro- or macro-average.

    The "nearest" split is found only within a limited window around the reference. Otherwise, you would compare splits
    that have nothing to do with each other. For example:
            Ref: AAA BBB CCC DDD
            Can: AAABBBCCC DDD
    A bad implementation will have two issues: it will compare the AAA BBB reference to the CCC DDD candidate and thus
    give score for a split that is too far away, and it will give score for a reference that is already aligned. The
    latter can be solved by using a 1-to-1 alignment algorithm, e.g. Viterbi on cumulative distance. The former issue
    the still exists: if you have
        Ref: AAAAA BBBBB CCCCC
        Can: AAAAABBBBB C CCCC
    An unwindowed aligner will ignore the fact that there is a perfect match of the B-C split and instead align it with the
    A-B reference, with the idea being that score(perfect B-C) < score(almost B-C) + score(poor A-B).
    This is not how you want your scoring to be, because we should not reward the tokeniser by pretending that the A-B
    split is represented successfully here.

    So, we only align splits within the span of (half of) surrounding morphs. Basically the same as above, except
    you don't necessarily need Viterbi (you can still do it with Viterbi, but just have the distance function drop to 0
    faster).
    If a split falls outside of the span, it should be aligned with a different reference split. For example:
        Ref: AAA BBB CCC
        Can: AAABB BCCC
    This counts as the AAA-BBB split being lost (0 score) and the BBB-CCC split having moved (partial score).
        Ref: AAA BBB CCC
        Can: AAAB BBCCC
    This counts as the AAA-BBB split having moved and the BBB-CCC split being lost.
    What should the partial score be? In the above example, for the last reference split:
        AAABBB CCC: score of 1.0
        AAABB BCCC: lower score because still aligned but not perfectly
        AAAB BBCCC: score of 0.0 because no longer aligned

     TODO: The current implementation does not use other candidate splits to determine the score of the chosen split,
           neither as a limit that shortens the distance scale nor as a tiebreaking step at the edges. Possibly you
           are forced to do Viterbi alignment with a window.

    Trade-off for this metric:
        - Pro: the converse metric also exists, making a natural precision-recall pair.
        - Con: Some ambiguity. Say that the last morph is very large, then what happens when the split moves right?
            AAABBB CCCCCCDDD: score of 1.0
            AAABBBC CCCCCDDD: lower score
            AAABBBCC CCCCDDD: lower score
            AAABBBCCC CCCDDD: lower score, although this is actually an edge case where the split could belong either to the morph boundary on the left or the right. I guess it doesn't matter in a sense, because both cases would have the same distance.
            AAABBBCCCC CCDDD: score of 0.0
          You could also wonder whether the distance function should be altered depending on other splits. For example:
            AAABBBCCC CCCDDD is ambiguous: does this split belong to B-C (distance 3) with missing C-D split, or is it the C-D split (distance 3) with missing B-C split?
            AAABBBCCC CCC DDD is strange: for each morph split, check the corresponding candidate split. The C-D split is found immediately. But what about the B-C split? There is indeed another split, but we did not know which reference it belonged to at first, yet now, it seems quite obvious that it should belong to the reference split that has no candidate yet. This is starting to look like an alignment/matching.
            AAABBBCCCCC C DDD is also strange: if aligning morphs, would the first split belong to nothing, or to the B-C reference split? It's out of range, but since the C-D split is correct, should you not get some score for at least trying to have the same amount of tokens as the reference?
            AAABBBCCCC C CDDD is even more strange: the last split clearly belongs to the C-D reference, but now, despite being out of range, does the B-C reference get to claim a candidate so far away as the first split?
            AAABBBCCC CCCD DD is equally strange: C-D is present off by 1, and B-C is either:
                1. missing entirely (aligning the first split with C-D as well);
                2. present and off by 3, the maximal distance before a score of 0 because CCCCCC has length 6, thus giving 1 - 3/(1+6//2) == 0.25 score.
                3. present and off by 3, but since C-D has been aligned already, even AAABBBCCCC CCD DD and AAABBBCCCCC CD DD could be considered as having a split in B-C range.
                   So then, the maximum distance is actually 5 or 6, not 3, thus yielding for the given split of distance 3 the score 1 - 3/(1+5) == 0.5 or 1 - 3/(1+6) == 0.57, not 0.25.

      A good conclusion of all of the above seems to be this:
        - You need injective alignment either way (one split matches at most one split, on both sides)
        - You need a distance function that is limited at the very least by the previous and next reference split (so a split can be at most one full morpheme off before getting score 0).
        - Optionally, you should limit by half of the distance to either. Optionally, you should limit further by other splits between the reference and you, but only those that have been meaningfully selected.

    :param adversarial: A version that, rather than choosing for each reference split the BEST candidate for it,
                        chooses the WORST candidate for it within the allowed range. The idea is that you make it as
                        hard as possible to get a perfect score and that every disastrous morpheme split should be counted.
    """
    lengths = list(map(len, morphs))
    starts = [0] + list(cumsum(lengths))[:-1]

    splits = list(cumsum(map(len, tokens)))[:-1]  # A split is represented as the index of the first character of the token after the split.
    split_index = 0
    n_splits = len(splits)

    for i in range(1, len(starts)):  # Iterate over all the morpheme boundaries.
        s_i = starts[i]  # Current morpheme boundary.
        l_left_half = lengths[i - 1] // 2  # Length of the left morpheme.
        l_right_half = lengths[i] // 2  # Length of the right morpheme.
        # print("Start:", s_i)

        # Advance splits until you get in range. (Only relevant for first splits, I believe.)
        while split_index < n_splits and splits[split_index] < s_i - l_left_half:
            split_index += 1

        # As long as the splits are within range, find the best one.
        best_score = 0 if not adversarial else 1
        while split_index < n_splits and splits[split_index] <= s_i + l_right_half:
            diff = splits[split_index] - s_i
            score = 1 - abs(diff) / (1 + (l_left_half if diff < 0 else l_right_half))
            if adversarial:
                best_score = min(best_score, score)
            else:
                best_score = max(best_score, score)

            # print("Diff", diff)
            split_index += 1

        output.add(best_score)

    output.fence()


########################################################################################################################


@dataclass
class ConfusionMatrices:
    cm:          ConfusionMatrix
    cm_weighted: ConfusionMatrix


class MorphologyIterable(ObservableRoot[Tuple[str,M]]):

    def __init__(self, experiment_id: str, dataset: ModestDataset[M], word_weights: Dict[str,float]=None, observers: List[Observer[Tuple[str,M]]]=None):
        super().__init__(cache_disambiguator=experiment_id, observers=observers)
        self._dataset = dataset
        self._weights = word_weights or dict()

    def _nodeIdentifier(self) -> str:
        return self._dataset.identifier()

    def _stream(self) -> Iterator[Tuple[Tuple[str,M],float]]:
        for obj in self._dataset.generate():
            word = obj.word
            yield (word, obj), self._weights.get(word, 1)


class MorphologyAsClassification(FinallyObservableObserver[Tuple[Tokens,M],ConfusionMatrices]):

    def __init__(self, visitor: MorphologyVisitor, effective_preprocessor: Preprocessor=None,
                 holdout: HoldoutState=None, do_log_false_negatives: bool=False,
                 observers: List[Observer[ConfusionMatrices]]=None):
        super().__init__(cache_disambiguator=visitor.__class__.__name__, observers=observers)
        self._visitor      = visitor
        self._preprocessor = effective_preprocessor
        self._holdout      = holdout
        self._do_log_fusions = do_log_false_negatives

    def _initialiseAsObserver(self, identifier: str):
        self._cm   = ConfusionMatrix()
        self._cm_w = ConfusionMatrix()
        self._log  = None if not self._do_log_fusions else open(TkTkTPaths.pathToEvaluations() / "" / f"{self._cacheIdentifier()}_morpheme-fusions.txt", "w", encoding="utf-8")

    def _receive(self, sample: Tuple[Tokens,M], weight: float):
        tokens, obj = sample
        tokeniser_segmentation = " ".join(self._preprocessor.undo_per_token(tokens)).strip()
        reference_segmentation = " ".join(self._visitor(obj))
        # print(reference_segmentation, "->", tokeniser_segmentation)

        # Compare
        tp, predicted, relevant, total = compareSplits_cursors(candidate=tokeniser_segmentation, reference=reference_segmentation)
        self._cm  .add(tp, predicted, relevant, total, weight=1)
        self._cm_w.add(tp, predicted, relevant, total, weight=weight)

        if self._do_log_fusions and tp != relevant:  # This condition means "if you merged somewhere you shouldn't have". It ignores errors of excess tokenisation (tp != predicted).
            self._log.write(reference_segmentation + "\t->\t" + tokeniser_segmentation + "\n")

    def _compute(self) -> ConfusionMatrices:
        if self._do_log_fusions:
            self._log.close()
        # if not quiet:
        #     self._cm.display()
        #     self._cm.displayRePrF1(indent=2)
        #     self._cm_w.display()
        #     self._cm_w.displayRePrF1(indent=2)
        return ConfusionMatrices(cm=self._cm, cm_weighted=self._cm_w)

    def _cachePath(self, unambiguous_cache_identifier: str) -> Path:
        return TkTkTPaths.extend(TkTkTPaths.pathToEvaluations(), ["morphology"]) / (unambiguous_cache_identifier + ".json")

    def _cacheStore(self, cache_path: Path, result: ConfusionMatrices):
        def matrixToDict(cm: ConfusionMatrix):
            tp, fp, tn, fn = cm.compute()
            return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump({"unweighted": matrixToDict(result.cm), "weighted": matrixToDict(result.cm_weighted)}, handle)

    def _cacheLoad(self, cache_path: Path) -> ConfusionMatrices:
        def dictToMatrix(d: dict) -> ConfusionMatrix:
            return ConfusionMatrix.fromPositivesNegatives(d["TP"], d["FP"], d["TN"], d["FN"])

        with open(cache_path, "r", encoding="utf-8") as handle:
            d = json.load(handle)
            return ConfusionMatrices(cm=dictToMatrix(d["unweighted"]), cm_weighted=dictToMatrix(d["weighted"]))
