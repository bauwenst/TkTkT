"""
Taken from the BPE knockout repo.
"""
from typing import Callable, Dict, Optional, List, Tuple
from dataclasses import dataclass

import re

from bpe_knockout.datahandlers.morphology import MorphologyVisitor, MorphSplit, LexSplit
from bpe_knockout.datahandlers.holdout import Holdout
from bpe_knockout.project.config import morphologyGenerator, lexiconWeights
from bpe_knockout.auxiliary.tokenizer_interface import tokenizeAsWord

from ..util.printing import wprint
from ..files.paths import getTkTkToutputPath
from ..interfaces.general import Tokeniser


# Segmentation kernel
SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))
def compareSplits(candidate: str, reference: str):
    """
    Takes two words split with spaces and computes the factors of the precision and recall of those splits.
    For example,
        candidate a bc d ef
        reference a b cd e f
    has precision 66% and recall 75%.

    Assumes they have the same amount of non-spaces.
    """
    c = " ".join(candidate.strip()).replace("   ", SPLIT_MARKER)
    r = " ".join(reference.strip()).replace("   ", SPLIT_MARKER)

    c_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(c)}
    r_indices = {match.start() for match in SPLIT_MARKER_RE.finditer(r)}

    tp = len(c_indices & r_indices)
    relevant = len(r_indices)
    predicted = len(c_indices)
    total = len(r) // 2
    return tp, predicted, relevant, total


def compareSplits2(candidate: str, reference: str):
    # Numpy implementation is short, but an order of magnitude slower!
    # c_indices = np.cumsum([len(t) for t in candidate.split()]) - 1
    # r_indices = np.cumsum([len(t) for t in reference.split()]) - 1
    c_indices = [len(t) for t in candidate.split()]
    r_indices = [len(t) for t in reference.split()]
    cum = 0
    for i in range(len(c_indices)):
        cum += c_indices[i]
        c_indices[i] = cum
    cum = 0
    for i in range(len(r_indices)):
        cum += r_indices[i]
        r_indices[i] = cum

    tp = len(set(c_indices) & set(r_indices)) - 1
    relevant = len(r_indices) - 1
    predicted = len(c_indices) - 1
    total = c_indices[-1] - 1
    return tp, predicted, relevant, total


def compareSplits3(candidate: str, reference: str):
    candidate_index = 0
    reference_index = 0

    tp        = 0
    relevant  = 0
    predicted = 0
    total     = 0
    while candidate_index < len(candidate):
        candidate_split = candidate[candidate_index] == " "
        reference_split = reference[reference_index] == " "

        tp += candidate_split and reference_split
        relevant += reference_split
        predicted += candidate_split
        total += 1

        candidate_index += 1 + candidate_split
        reference_index += 1 + reference_split

    return tp, predicted, relevant, total - 1


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

    def compute(self):
        precision = self.total_tp/self.total_predicted if self.total_predicted else 1.0
        recall    = self.total_tp/self.total_relevant  if self.total_relevant  else 1.0
        f1        = ConfusionMatrix.f1(precision, recall)
        return precision, recall, f1

    def display(self):
        N  = self.total
        tp = self.total_tp
        fp = self.total_predicted - self.total_tp
        fn = self.total_relevant - self.total_tp
        tn = N - tp - fp - fn

        string = "        \tpredicted\n"    +\
                 "        \t  +  \t  -\n"   +\
                f"actual +\t {tp}\t {fn}\n" +\
                f"       -\t {fp}\t {tn}"
        wprint(string)

    def computeAndDisplay(self, indent=0):
        P, R, F1 = self.compute()
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

        tuples = [matrix.compute() for matrix in matrices]
        precisions, recalls, f1s = zip(*tuples)
        return sum(precisions)/n, sum(recalls)/n, sum(f1s)/n


#########################
### Testing framework ###
#########################
def morphologyVersusTokenisation(morphology_method: MorphologyVisitor, tokenizer=robbert_tokenizer,  # Compared
                                 weights: Dict[str, float]=None, holdout: Holdout=None,  # Experimental parameters
                                 do_write_errors=False, quiet=False, display_confusion_matrix=False, log_name="log"):  # Display
    # Optional stuff
    weighted = weights is not None
    if do_write_errors:
        output_dir = getTkTkToutputPath() / "evaluation"
        output_dir.mkdir(exist_ok=True)
        log = open(output_dir / f"{log_name}_boundary_violations_{morphology_method.__name__}.txt", "w", encoding="utf-8")

    cm   = ConfusionMatrix()
    cm_w = ConfusionMatrix() if weighted else None

    if holdout is None:
        holdout = Holdout(0.0)  # 0% is in the training set, 100% in the test set.

    for obj in holdout(morphologyGenerator(verbose=not quiet), test=True):
        lemma = obj.lemma()

        tokeniser_segmentation = " ".join(tokenizeAsWord(lemma, tokenizer=tokenizer)).strip()
        reference_segmentation = morphology_method(obj)

        # Compare
        tp, predicted, relevant, total = compareSplits(candidate=tokeniser_segmentation, reference=reference_segmentation)
        cm.add(tp, predicted, relevant, total, 1)
        if weighted:
            cm_w.add(tp, predicted, relevant, total, weight=weights.get(lemma, 1))

        # The .write condition below is a bit too sensitive w.r.t. interfices and prepositions [P] or adverbs [B].
        # Perhaps need to compare against two acceptable splits? OTOH, splitting off an interfix is healthy
        # since you're not duplicating a subword.
        # Examples:
        #     bruid s nacht     tokenised as	bruids nacht
        #     voet bal match	tokenised as	voetbal match
        #     vlieg en papier   tokenised as	vliegen papier
        #     bouw toe zicht    tokenised as	bouw toezicht
        #     voor hoofd        tokenised as	voorhoofd
        #     weg nemen         tokenised as	wegnemen
        #     wiel er baan      tokenised as	wieler baan
        if do_write_errors and tp != relevant:  # This condition means "if you merged somewhere you shouldn't have". It ignores errors of excess tokenisation (tp != predicted).
            log.write(reference_segmentation + "\t->\t" + tokeniser_segmentation + "\n")

    if do_write_errors:
        log.close()

    if not quiet:
        # Pr, Re, F1
        cm.computeAndDisplay(indent=2)
        if weighted:
            print("\tWeighted:")
            cm_w.computeAndDisplay(indent=2)

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
def intrinsicEvaluation(tokenisers: List[BasicStringTokeniser],
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
        wprint(f"Batch evaluation of {len(tokenisers)} tokenisers...")

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
        cm1, cm1_w = morphologyVersusTokenisation(MorphSplit(), tokenizer=t,
                                                  weights=lemma_weights, holdout=holdout,
                                                  do_write_errors=False, log_name=name, quiet=not verbose)

        if do_whole_word:
            if verbose:
                wprint("\tLemmatic split accuracy:")
            cm2, cm2_w = morphologyVersusTokenisation(LexSplit(), tokenizer=t,
                                                      weights=lemma_weights, holdout=holdout,
                                                      do_write_errors=False, log_name=name, quiet=not verbose)
        else:
            cm2, cm2_w = None, None
        print()

        results.append(TokeniserEvaluation(name=name,
                                           cm_morph=cm1, cm_morph_w=cm1_w,
                                           cm_lex=cm2,   cm_lex_w=cm2_w))
    return results
