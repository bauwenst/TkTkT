"""
Taken from the BPE knockout repo.
"""
from typing import Callable, Dict, List, Tuple, Iterable, Union
from dataclasses import dataclass

from bpe_knockout.datahandlers.holdout import Holdout
from bpe_knockout.project.config import morphologyGenerator, lexiconWeights
from modest.interfaces.morphologies import MorphologyVisitor, WordSegmentation, MorphSplit, FreeMorphSplit

from ..util.printing import wprint
from ..util.aggregates import ConfusionMatrix
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
