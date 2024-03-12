"""
Taken from the BPE knockout repo.
"""
from typing import Callable, Dict, Optional, List, Tuple, Iterable
from dataclasses import dataclass

from bpe_knockout.datahandlers.morphology import MorphologyVisitor, MorphSplit, LexSplit, LemmaMorphology
from bpe_knockout.datahandlers.holdout import Holdout
from bpe_knockout.project.config import morphologyGenerator, lexiconWeights

from ..util.printing import wprint
from ..files.paths import DataPaths
from ..interfaces.tokeniser import Tokeniser


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


class ConfusionMatrix:

    def __init__(self):
        self.total_tp        = 0
        self.total_predicted = 0
        self.total_relevant  = 0
        self.total           = 0

    def add(self, tp: int, predicted: int, relevant: int, total: int, weight: float=1):
        self.total_tp        += weight*tp
        self.total_predicted += weight*predicted
        self.total_relevant  += weight*relevant
        self.total           += weight*total

    def computePrReF1(self):
        precision = self.total_tp/self.total_predicted if self.total_predicted else 1.0
        recall    = self.total_tp/self.total_relevant  if self.total_relevant  else 1.0
        f1        = ConfusionMatrix.f1(precision, recall)
        return precision, recall, f1

    def compute(self):
        N  = self.total
        tp = self.total_tp
        fp = self.total_predicted - self.total_tp
        fn = self.total_relevant - self.total_tp
        tn = N - tp - fp - fn
        return tp, fp, tn, fn

    def display(self):
        tp, fp, tn, fn = self.compute()
        string = "        \tpredicted\n"    +\
                 "        \t  +  \t  -\n"   +\
                f"actual +\t {tp}\t {fn}\n" +\
                f"       -\t {fp}\t {tn}"
        wprint(string)

    def displayRePrF1(self, indent=0):
        P, R, F1 = self.computePrReF1()
        wprint("\t"*indent + "Precision:", P)
        print("\t"*indent + "Recall:   ", R)
        print("\t"*indent + "F1:       ", F1)

    @staticmethod
    def f1(precision: float, recall: float):
        return 2*(precision*recall)/(precision+recall)

    @staticmethod
    def computeMatrixMacroAverage(matrices: List["ConfusionMatrix"]) -> Tuple[float, float, float]:
        """
        Computes the macro-average Pr, Re, F1 for a list of confusion matrices.

        Note: although the Pr, Re, F1 returned by .compute() are a micro-average, this method is not the macro-average
        equivalent of that. This is because .compute() is the micro-average over all added word segmentations, NOT over
        a list of matrices. It is impossible to reconstruct the macro-average over word segmentations because we don't store
        their separate Pr, Re, F1.
        """
        n = len(matrices)
        if n == 0:
            return (1.0, 1.0, 1.0)

        tuples = [matrix.computePrReF1() for matrix in matrices]
        precisions, recalls, f1s = zip(*tuples)
        return sum(precisions)/n, sum(recalls)/n, sum(f1s)/n


#########################
### Testing framework ###
#########################
def tokeniseAndDecode(string: str, tokeniser: Tokeniser) -> List[str]:
    """
    Tokenisation, but afterwards, you run each produced token back through (inverse of) the pretokeniser and normaliser.
    """
    return tokeniser.preprocessor.undo_per_token(tokeniser.prepareAndTokenise(string))


def morphologyVersusTokenisation(
        morphological_generator: Iterable[LemmaMorphology], morphology_method: MorphologyVisitor,
        tokeniser: Tokeniser,
        weights: Dict[str, float]=None, holdout: Holdout=None,  # Experiment parameters
        do_write_fusions: bool=False, quiet: bool=False, display_confusion_matrix: bool=False, log_name: str="log"  # Display
    ) -> Tuple[ConfusionMatrix, ConfusionMatrix]:
    # Optional stuff
    weighted = weights is not None
    if do_write_fusions:
        output_dir = DataPaths.pathToEvaluations()
        log = open(output_dir / f"{log_name}_morpheme-fusions_{morphology_method.__name__}.txt", "w", encoding="utf-8")

    # Result storage
    cm   = ConfusionMatrix()
    cm_w = ConfusionMatrix() if weighted else None

    if holdout is None:
        holdout = Holdout(0.0)  # 0% is in the training set, 100% in the test set.

    for obj in holdout(morphological_generator, test=True):
        lemma = obj.lemma()

        tokeniser_segmentation = " ".join(tokeniseAndDecode(lemma, tokeniser=tokeniser)).strip()
        reference_segmentation = morphology_method(obj)

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

    cm_morph:   ConfusionMatrix
    cm_morph_w: ConfusionMatrix
    cm_lex:     ConfusionMatrix
    cm_lex_w:   ConfusionMatrix


# @timeit
def intrinsicEvaluation(tokenisers: Iterable[Tokeniser],
                        reweighting_function: Callable[[float], float]=None, holdout: Holdout=None, do_whole_word=False,
                        verbose=False) -> List[TokeniserEvaluation]:
    """
    Generates, for each given tokeniser, 12 metrics:
        - Morph split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;
        - Lemmatic (whole-word) split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;

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
                                                      LexSplit(), tokeniser=t,
                                                      weights=lemma_weights, holdout=holdout,
                                                      do_write_fusions=False, log_name=name, quiet=not verbose)
        else:
            cm2, cm2_w = None, None
        print()

        results.append(TokeniserEvaluation(name=name,
                                           cm_morph=cm1, cm_morph_w=cm1_w,
                                           cm_lex=cm2,   cm_lex_w=cm2_w))
    return results
