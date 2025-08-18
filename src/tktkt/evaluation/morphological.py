"""
Taken from the BPE knockout repo.
"""
from typing import Callable, Dict, List, Tuple, Iterable, Union
from dataclasses import dataclass

from bpe_knockout.datahandlers.holdout import Holdout
from bpe_knockout.project.config import morphologyGenerator, lexiconWeights
from modest.interfaces.morphologies import MorphologyVisitor, WordSegmentation, MorphSplit, FreeMorphSplit

from ..util.printing import wprint
from ..util.aggregates import ConfusionMatrix, NestedAverage, NestedMicroMacro
from ..util.iterables import cumsum
from ..paths import TkTkTPaths
from ..interfaces.tokeniser import Tokeniser, TokeniserWithFiniteIdRange


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


#########################
### Testing framework ###
#########################
def tokeniseAndDecode(string: str, tokeniser: Tokeniser) -> List[str]:
    """
    Tokenisation, but afterwards, you run each produced token back through (inverse of) the pretokeniser and (invertible) mappings.
    """
    return tokeniser.preprocessor.undo_per_token(tokeniser.prepareAndTokenise(string))


def morphologyVersusTokenisation(
        morphological_generator: Iterable[WordSegmentation], morphology_method: MorphologyVisitor,
        tokeniser: Tokeniser,
        weights: Dict[str, float]=None, holdout: Holdout=None,  # Experiment parameters
        do_write_fusions: bool=False, quiet: bool=False, display_confusion_matrix: bool=False, log_name: str="log"  # Display
    ) -> Tuple[ConfusionMatrix, ConfusionMatrix]:
    # Optional stuff
    weighted = weights is not None
    log = None
    if do_write_fusions:
        log = open(TkTkTPaths.pathToEvaluations() / f"{log_name}_morpheme-fusions_{morphology_method.__name__}.txt", "w", encoding="utf-8")

    # Result storage
    cm   = ConfusionMatrix()
    cm_w = ConfusionMatrix() if weighted else None

    if holdout is None:
        holdout = Holdout(0.0)  # 0% is in the training set, 100% in the test set.

    for obj in holdout(morphological_generator, test=True):
        lemma = obj.word

        tokeniser_segmentation = " ".join(tokeniseAndDecode(lemma, tokeniser=tokeniser)).strip()
        reference_segmentation = " ".join(morphology_method(obj))

        # print(reference_segmentation, "->", tokeniser_segmentation)

        # Compare
        tp, predicted, relevant, total = compareSplits_cursors(candidate=tokeniser_segmentation, reference=reference_segmentation)
        cm.add(tp, predicted, relevant, total, 1)
        if weighted:
            cm_w.add(tp, predicted, relevant, total, weight=weights.get(lemma, 1))

        if do_write_fusions and tp != relevant:  # This condition means "if you merged somewhere you shouldn't have". It ignores errors of excess tokenisation (tp != predicted).
            log.write(reference_segmentation + "\t->\t" + tokeniser_segmentation + "\n")

    if do_write_fusions:
        log.close()

    if not quiet:
        # Pr, Re, F1
        cm.displayRePrF1(indent=2)
        if weighted:
            print("\tWeighted:")
            cm_w.displayRePrF1(indent=2)

        # Confusion matrices (TP, FP, FN, TN).
        if display_confusion_matrix:
            print("Confusion matrix:")
            cm.display()
            if weighted:
                print("Weighted confusion matrix:")
                cm_w.display()

    return cm, cm_w


@dataclass
class TokeniserEvaluation:
    name: str
    vocabsize: int

    cm_morph:   ConfusionMatrix
    cm_morph_w: ConfusionMatrix
    cm_lex:     ConfusionMatrix
    cm_lex_w:   ConfusionMatrix


# @timeit
def intrinsicEvaluation(tokenisers: Iterable[Union[Tokeniser, TokeniserWithFiniteIdRange]],
                        reweighting_function: Callable[[float], float]=None, holdout: Holdout=None, do_whole_word=False,
                        verbose=False) -> List[TokeniserEvaluation]:
    """
    Generates, for each given tokeniser, 12 metrics:
        - Morph-level unweighted and weighted precision, recall, F1 of morphological split positions;
        - Whole-word unweighted and weighted precision, recall, F1 of split positions;

    Uses the morphology file (for both) and lemma weights (for the latter) in the CURRENTLY ACTIVE KnockoutDataContext.

    :param tokenisers: The elements of the given list must have a method .tokenize(str) -> List[str].
    :param reweighting_function: Applied to lemma frequencies. If no function is given, the weighted metrics are dropped
                                 (rather than applying the identity function to the frequencies).
                                 It's useful to not automatically fill this function in, because the reweighting function
                                 used in the config is used in BTE training and nobody says that it needs to be equal here.
    """
    if verbose:
        wprint(f"Batch evaluation of {len(tokenisers) if isinstance(tokenisers, (list, tuple)) else 'generated'} tokenisers...")

    # Load weights
    lemma_weights = lexiconWeights(reweighting_function) if reweighting_function is not None else None  # If it is None, this is used as a signal to say "I don't want weighting".

    # Evaluation loop
    results = []
    for t in tokenisers:
        # Get metadata
        try:
            name = t.getName()
        except:
            name = t.__class__.__name__

        try:
            vocabsize = t.getVocabSize()
        except:
            vocabsize = 0  # Technically not wrong, although infinity is equally true.

        # Uncomment this if you need to only simulate the testing framework (e.g. calling this test in a loop), rather than get results.
        # results.append(TokeniserEvaluation(name=name, vocabsize=size, cm_morph=SegmentationConfusionMatrix(), cm_morph_w=SegmentationConfusionMatrix(), cm_lex=SegmentationConfusionMatrix(), cm_lex_w=SegmentationConfusionMatrix()))
        # continue

        # Print and evaluate
        if verbose:
            print(name)
            wprint("\tMorph split accuracy:")
        cm1, cm1_w = morphologyVersusTokenisation(morphologyGenerator(verbose=verbose),
                                                  MorphSplit(), tokeniser=t,
                                                  weights=lemma_weights, holdout=holdout,
                                                  do_write_fusions=False, log_name=name, quiet=not verbose)

        if do_whole_word:
            if verbose:
                wprint("\tLemmatic split accuracy:")
            cm2, cm2_w = morphologyVersusTokenisation(morphologyGenerator(verbose=verbose),
                                                      FreeMorphSplit(), tokeniser=t,
                                                      weights=lemma_weights, holdout=holdout,
                                                      do_write_fusions=False, log_name=name, quiet=not verbose)
        else:
            cm2, cm2_w = None, None

        if verbose:
            wprint()

        results.append(TokeniserEvaluation(name=name, vocabsize=vocabsize,
                                           cm_morph=cm1, cm_morph_w=cm1_w,
                                           cm_lex=cm2,   cm_lex_w=cm2_w))
    return results
