from src.tktkt.models.viterbi.framework import *
from src.tktkt.models.viterbi.accumulators import *
from src.tktkt.models.viterbi.objectives_unguided import *


def makeTokeniser():
    from transformers import RobertaTokenizer
    baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    vocab = baseline.get_vocab()

    objective1 = ViterbiObjective(initial_score=0, score_generator=MinimiseTokenAmount(vocab), score_combiner=Plus())
    objective2 = ViterbiObjective(initial_score=0, score_generator=MaximiseTokenLength(vocab), score_combiner=Max())

    # max_step = max(map(len,vocab.keys()))
    max_step = 15
    print("K:", max_step)

    from src.tktkt.preparation.splitters import WordSplitter
    from src.tktkt.preparation.spacemarking import ROBERTA_SPACING

    return ViterbiTokeniser(WordSplitter(ROBERTA_SPACING), max_step, [objective1, objective2])


if __name__ == "__main__":
    tk = makeTokeniser()
    print(tk.prepareAndTokenise(" flatscreentelevisie"))
