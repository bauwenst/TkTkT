"""
Taken from the BPE knockout repo.
"""
from typing import Callable, Dict, Optional, List, Tuple
from dataclasses import dataclass

import re

from bpe_knockout.datahandlers.morphology import MorphSplit, LexSplit  # TODO: Should not always do LexSplit evaluation because not all morphological corpora have it.
from bpe_knockout.datahandlers.wordfiles import *
from bpe_knockout.datahandlers.holdout import Holdout
from bpe_knockout.project.paths import *
from bpe_knockout.project.config import morphologyGenerator, lexiconWeights
from bpe_knockout.auxiliary.robbert_tokenizer import robbert_tokenizer
from bpe_knockout.auxiliary.tokenizer_interface import tokenizeAsWord, BasicStringTokeniser

from ..util.printing import wprint


# Segmentation kernel  TODO: I wonder if you can make this faster, e.g. with split() and len() rather than regex.
SPLIT_MARKER = "|"
SPLIT_MARKER_RE = re.compile(re.escape(SPLIT_MARKER))

class SegmentationConfusionMatrix:

    def __init__(self):
        self.total_tp = 0
        self.total_predicted = 0
        self.total_relevant = 0
        self.total = 0

    def add(self, candidate: str, reference: str, weight: float=1):
        tp, predicted, relevant, total = SegmentationConfusionMatrix.compareSplits(candidate, reference)  # TODO Should not be run twice for weighted vs unweighted
        self.total_tp        += weight*tp
        self.total_predicted += weight*predicted
        self.total_relevant  += weight*relevant
        self.total           += weight*total
        return tp, predicted, relevant, total

    def compute(self):
        precision = self.total_tp/self.total_predicted if self.total_predicted else 1.0
        recall    = self.total_tp/self.total_relevant  if self.total_relevant  else 1.0
        f1        = SegmentationConfusionMatrix.f1(precision, recall)
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

        tp        = len(c_indices & r_indices)
        relevant  = len(r_indices)
        predicted = len(c_indices)
        total     = len(r)//2
        return tp, predicted, relevant, total

    @staticmethod
    def f1(precision: float, recall: float):
        return 2*(precision*recall)/(precision+recall)

    @staticmethod
    def computeMatrixMacroAverage(matrices: List["SegmentationConfusionMatrix"]) -> Tuple[float, float, float]:
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
        log = open(PATH_DATA_OUT / f"{log_name}_boundary_violations_{morphology_method.__name__}.txt", "w", encoding="utf-8")

    cm   = SegmentationConfusionMatrix()
    cm_w = SegmentationConfusionMatrix() if weighted else None

    if holdout is None:
        holdout = Holdout(0.0)  # 0% is in the training set, 100% in the test set.

    for obj in holdout(morphologyGenerator(verbose=not quiet), test=True):
        lemma = obj.lemma()

        # Get space-segmented word from the tokeniser.
        bpe_segmentation = " ".join(tokenizeAsWord(lemma, tokenizer=tokenizer)).strip()

        # Generate lexemic split
        reference_segmentation = morphology_method(obj)

        # Compare
        tp, _, relevant, _ = cm.add(candidate=bpe_segmentation, reference=reference_segmentation)
        if weighted:
            amplification = weights.get(lemma, 1)
            cm_w.add(candidate=bpe_segmentation, reference=reference_segmentation, weight=amplification)  # TODO: This is slow. You are re-checking all the split positions, just to get the same exact TP, FP, FN, TN, which are just weighted differently.

        # FIXME: The precision and recall are fine, but the .write condition below is a bit too sensitive at the
        #        moment w.r.t. interfices and prepositions [P] or adverbs [B]. Perhaps need to allow two lexeme
        #        splits? OTOH, splitting off an interfix is healthy since you're not duplicating a subword.
        #        Examples:
        #           bruid s nacht	tokenised as	bruids nacht
        #           voet bal match	tokenised as	voetbal match
        #           vlieg en papier	tokenised as	vliegen papier
        #           bouw toe zicht	tokenised as	bouw toezicht
        #           voor hoofd	    tokenised as	voorhoofd
        #           weg nemen	    tokenised as	wegnemen
        #           wiel er baan    tokenised as	wieler baan
        if do_write_errors and tp != relevant:  # This condition means "if you merged somewhere you shouldn't have". It ignores errors of excess tokenisation (tp != predicted).
            log.write(reference_segmentation + "\t->\t" + bpe_segmentation + "\n")

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
    vocabsize: int
    cm_morph: SegmentationConfusionMatrix
    cm_morph_w: SegmentationConfusionMatrix
    cm_lex: SegmentationConfusionMatrix
    cm_lex_w: SegmentationConfusionMatrix


# @timeit
def test_tokenizers_batch(tkzrs: List[BasicStringTokeniser], reweighting_function: Callable[[float], float]=None, holdout: Holdout=None) -> List[TokeniserEvaluation]:
    """
    Generates, for each given tokeniser, 12 metrics:
        - Morph split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;
        - Lemmatic split unweighted and weighted precision, recall, F1 of split positions vs. e-Lex;

    :param tkzrs: The elements of the given list must have a method .tokenize(str) -> List[str].
    :param reweighting_function: Applied to lemma frequencies. If no function is given, the weighted metrics are dropped
                                 (rather than applying the identity function to the frequencies).
                                 It's useful to not automatically fill this function in, because the reweighting function
                                 used in the config is used in BTE training and nobody says that it needs to be equal here.
    """
    wprint(f"Batch evaluation of {len(tkzrs)} tokenisers...")

    # Load weights
    lemma_weights = lexiconWeights(reweighting_function) if reweighting_function is not None else None  # If it is None, this is used as a signal to say "I don't want weighting".

    # Evaluation loop
    results = []
    for t in tkzrs:
        # Get metadata
        try:
            name = t.getName()
        except:
            name = t.__class__.__name__

        size = t.vocab_size

        # Uncomment this if you need to only simulate the testing framework, rather than get results.
        # results.append(TokeniserEvaluation(name=name, vocabsize=size, cm_morph=SegmentationConfusionMatrix(), cm_morph_w=SegmentationConfusionMatrix(), cm_lex=SegmentationConfusionMatrix(), cm_lex_w=SegmentationConfusionMatrix()))
        # continue

        # Print and evaluate
        print(name)
        print("|V|:", size)
        wprint("\tMorph split accuracy:")
        cm1, cm1_w = morphologyVersusTokenisation(MorphSplit(), tokenizer=t, do_write_errors=False, log_name=name,
                                                  weights=lemma_weights, holdout=holdout)

        wprint("\tLemmatic split accuracy:")
        cm2, cm2_w = morphologyVersusTokenisation(LexSplit(), tokenizer=t, do_write_errors=False, log_name=name,
                                                  weights=lemma_weights, holdout=holdout)
        print()

        results.append(TokeniserEvaluation(name=name, vocabsize=size,
                                           cm_morph=cm1, cm_morph_w=cm1_w,
                                           cm_lex=cm2, cm_lex_w=cm2_w))
    return results
