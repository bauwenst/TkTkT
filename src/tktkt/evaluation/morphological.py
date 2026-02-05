"""
Taken from the BPE knockout repo.
"""
from typing import Self
from dataclasses import dataclass

import json

from modest.interfaces.morphologies import MorphologyVisitor, MorphSplit, FreeMorphSplit
from modest.interfaces.datasets import ModestDataset, M

from ..util.aggregates import ConfusionMatrix, NestedAverage, NestedMicroMacro
from ..util.interfaces import Cacheable, C
from ..util.iterables import cumsum
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


def morphIntegrity(morphs: list[str], tokens: list[str], output: NestedMicroMacro):
    """
    For each morph, find the token which contains most of its characters and compute the fraction of its total length
    that is present in that token. Then micro- or macro-average this across morphs in the corpus.
    Note that tokens are allowed to be the majority token for more than one morph. In the most extreme case: if the entire
    word is represented by one token, then all the morphs have 100% integrity.

    Given is the reference AAA BBB CCC. When treating morphological alignment as binary classification,
    a candidate AAA BBBC CC shows up the same as AAA BBBCC C (both have two splits, of which one is precise
    and both recall one of two splits). Meanwhile, we want to have a metric that measures the fact that
    the second one's token dedicated to the last morpheme is much less informative than the first one's,
    because most of the morpheme has disappeared.

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


def morphDilution(morphs: list[str], tokens: list[str], token_turbidity: NestedAverage, morph_dilution: NestedAverage, morph_muddiness: NestedAverage):
    """
    I have what I call the "morphological dilution hypothesis", which is slightly different from the morphological
    alignment hypothesis: language models perform worse in interpreting a morpheme when a strict substring of that morpheme
    appears in a token that carries any information about at least one other morpheme, and equivalently, language models
    perform worse in interpreting a token when it contains information from multiple morphemes of which at least one
    a strict substring.

    We define a contaminated token as "a token containing the characters of more than one morph and for at least one
    of those morphs is missing a character".
    We define a contaminated morph as "a morph for which there exists at least one contaminated token where that morph
    is missing a character".

    The rate of token contamination can be called token contamination or token turbidity.
    The rate of morphs that are partially concatenated to other morphs can be called the morph dilution.
    The rate of morphs which have a partial morph concatenated to them can be called the morph contamination or morph turbidity.
    The rate of morphs which are either partially concatenated to other morphs, or have a partial morph concatenated to
    them, can be called morph muddiness or morph confusion.

    The morph metrics are roughly interpretable as boundary false-positive rate (i.e. rate of splitting in the wrong
    place), except a wrong split is ignored if it is bordered by another split that isolates it from the adjacent morphs.
    """
    contaminated_tokens = []
    contaminated_morphs = set()
    diluted_morphs      = set()

    morph_splits = list(cumsum(map(len, morphs)))
    token_splits = list(cumsum(map(len, tokens)))

    current_morph = 0
    current_morph_end = morph_splits[current_morph]

    starts_initial_morph = True
    for token_id, index in enumerate(token_splits):
        initial_morph = current_morph
        current_morph_already_started = True
        while index >= current_morph_end:  # Iterate over all morphs that were finished by this token. (The morph being handled in the loop body is the one that currently ends at current_morph_end.)

            # End of loop increments
            current_morph_already_started = index != current_morph_end  # We know that when the current morph's boundary == index, the next morph won't have started by the end of this token.
            current_morph += 1
            if current_morph < len(morph_splits):
                current_morph_end = morph_splits[current_morph]
            else:
                current_morph_end = float("inf")

        # The information you now want to figure out is
        #    1. Is the left contaminating? (if yes, it is initial_morph)
        #    2. Is the right contaminating? (if yes, it is current_morph)
        #    3. If there is contamination coming from the left, does it include or exclude current_morph?
        contamination_from_left  = not starts_initial_morph and initial_morph != current_morph and (current_morph_already_started or current_morph > initial_morph + 1)  # Basically, we are just trying to avoid the case A|AA|BBB, the first time initial != current yet A does not contaminate current. So we introduce 'not current_morph_already_started', but then A|AABBB|CCC is excluded, so we introduce the last check.
        contamination_from_right = initial_morph != current_morph and current_morph_already_started

        if contamination_from_left:
            diluted_morphs.add(initial_morph)
            for i in range(initial_morph+1, current_morph+1 - (not current_morph_already_started)):
                contaminated_morphs.add(i)
        if contamination_from_right:
            diluted_morphs.add(current_morph)
            for i in range(initial_morph, current_morph):
                contaminated_morphs.add(i)
        if contamination_from_left or contamination_from_right:
            contaminated_tokens.append(token_id)

        # Deprecated: Boolean equivalent, but less interpretable way to check if a token is contaminated.
        # if starts_initial_morph:  # I.e. the initial morph you're seeing is NOT part of a preceding token. So, if it finishes within this token, then it does NOT count as contamination.
        #     if initial_morph != current_morph and current_morph_already_started:  # First check excludes first token in AAA|AAA|BBB (i.e. you're allowed to start a morph and not end it if that same morph keeps going afterwards), second check excludes first token in AAA|BBB and AAABBB|CCC (i.e. you're allowed to start a morph if you also end it).
        #         contaminated_tokens.append(token_id)
        # else:
        #     if initial_morph != current_morph and (current_morph_already_started or current_morph > initial_morph + 1):  # First check excludes the second token in AAA|AAA|ABBB (i.e. you're allowed to not start a morph if you also don't end it), second check excludes the second token in AAA|AAA|BBB (i.e. you're allowed to not start a morph and still end it, but only if you don't already start the next morph) and third check re-includes the second token in AAA|AAABBB|CCC (i.e. even though you're allowed to not start a morph if you end it cleanly, what must be ending cleanly is that morph, not another one).
        #         contaminated_tokens.append(token_id)

        # The next token starts its first morph if the current morph does not include its final morph.
        starts_initial_morph = not current_morph_already_started

    # print("     Bad tokens:", [tokens[i] for i in contaminated_tokens])
    # print(" Diluted morphs:", [morphs[i] for i in sorted(diluted_morphs)])
    # print("Impacted morphs:", [morphs[i] for i in sorted(contaminated_morphs)])

    token_turbidity.addMany(len(contaminated_tokens),                  len(tokens))
    morph_dilution .addMany(len(diluted_morphs),                       len(morphs))
    morph_muddiness.addMany(len(diluted_morphs | contaminated_morphs), len(morphs))
    token_turbidity.fence()
    morph_dilution .fence()
    morph_muddiness.fence()


def alignedSegmentationScore(morphs: list[str], tokens: list[str], adversarial: bool, output: NestedAverage):
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

        output.addOne(best_score)

    output.fence()


########################################################################################################################


@dataclass
class ConfusionMatrices(Cacheable):
    cm:          ConfusionMatrix
    cm_weighted: ConfusionMatrix

    _FILENAME = "confusion-matrices.json"

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return (cache_path / ConfusionMatrices._FILENAME).exists()

    def store(self, cache_path: Path):
        def matrixToDict(cm: ConfusionMatrix):
            tp, fp, tn, fn = cm.compute()
            return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

        with open(cache_path / ConfusionMatrices._FILENAME, "w", encoding="utf-8") as handle:
            json.dump({"unweighted": matrixToDict(self.cm), "weighted": matrixToDict(self.cm_weighted)}, handle)

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        def dictToMatrix(d: dict) -> ConfusionMatrix:
            return ConfusionMatrix.fromPositivesNegatives(d["TP"], d["FP"], d["TN"], d["FN"])

        with open(cache_path / ConfusionMatrices._FILENAME, "r", encoding="utf-8") as handle:
            d = json.load(handle)
            return ConfusionMatrices(cm=dictToMatrix(d["unweighted"]), cm_weighted=dictToMatrix(d["weighted"]))


class MorphologyIterable(ObservableRoot[tuple[str,M]]):

    def __init__(self, experiment_id: str, dataset: ModestDataset[M], word_weights: dict[str,float]=None, observers: list[Observer[tuple[str,M]]]=None):
        super().__init__(disambiguator=experiment_id, observers=observers)
        self._dataset = dataset
        self._weights = word_weights or dict()

    def _identifierPartial(self) -> str:
        return self._dataset.identifier()

    def _stream(self) -> Iterator[tuple[tuple[str,M],float]]:
        for obj in self._dataset.generate():
            word = obj.word
            yield (word, obj), self._weights.get(word, 1)


class MorphologyAsClassification(FinallyObservableObserver[tuple[Tokens,M],ConfusionMatrices]):

    def __init__(self, visitor: MorphologyVisitor, effective_preprocessor: Preprocessor,
                 do_log_false_negatives: bool=False,
                 observers: list[Observer[ConfusionMatrices]]=None):
        super().__init__(observers=observers)
        self._visitor      = visitor
        self._preprocessor = effective_preprocessor
        self._do_log_fusions = do_log_false_negatives

    def _identifierPartial(self) -> str:
        return self._visitor.__class__.__name__

    def _cacheType(self):
        return ConfusionMatrices

    def _cacheSubfolders(self) -> list[str]:
        return ["morphology"]

    def _initialiseAsObserver(self, parent_observable_identifier: str):
        self._cm   = ConfusionMatrix()
        self._cm_w = ConfusionMatrix()
        self._log  = None if not self._do_log_fusions else open(self._cachePath(parent_observable_identifier) / "morpheme-fusions.txt", "w", encoding="utf-8")

    def _receive(self, sample: tuple[Tokens,M], weight: float):
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


class ConfusionMatrixSummary(ImmediatelyObservableObserver[ConfusionMatrices,dict]):

    def _identifierPartial(self) -> str:
        return ""

    def _transit(self, sample: ConfusionMatrices, weight: float) -> dict:
        pr, re, f1       = sample.cm.computePrReF1()
        pr_w, re_w, f1_w = sample.cm_weighted.computePrReF1()
        return {
            "pr": pr,
            "re": re,
            "f1": f1,
            "pr_w": pr_w,
            "re_w": re_w,
            "f1_w": f1_w,
        }
