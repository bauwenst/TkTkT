from tktkt.models.viterbi.framework import *
from tktkt.models.viterbi.accumulators import *
from tktkt.models.viterbi.objectives_unguided import *
from tktkt.models.viterbi.objectives_postprocessors import *

def makeTokeniser():
    from transformers import RobertaTokenizer
    baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    vocab = baseline.get_vocab()

    objective1 = ViterbiObjective(initial_score=0, score_generator=ConstrainVocabulary(ConstantScore(), vocab, reset_value=-INFTY), score_combiner=ScoreSum())
    objective2 = ViterbiObjective(initial_score=0, score_generator=ConstrainVocabulary(TokenLength(), vocab, reset_value=-INFTY), score_combiner=ScoreMax())

    # max_step = max(map(len,vocab.keys()))
    max_step = 15
    print("K:", max_step)

    from src.tktkt.preparation.splitters import RobertaPretokeniser
    return ViterbiTokeniser(RobertaPretokeniser, max_step, [objective1, objective2])


if __name__ == "__main__":
    tk = makeTokeniser()
    print(tk.prepareAndTokenise(" flatscreentelevisie"))
