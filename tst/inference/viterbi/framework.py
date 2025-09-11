from tktkt.models.predictive.viterbi import *

def makeTokeniser():
    from transformers import RobertaTokenizer
    baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    vocab = baseline.get_vocab()

    objective1 = ViterbiObjective(initial_score=0, score_generator=VocabularyConstraintExact(ConstantScore(), vocab, reset_value=-INFTY), score_combiner=ScoreSum())
    objective2 = ViterbiObjective(initial_score=0, score_generator=VocabularyConstraintExact(TokenLength(), vocab, reset_value=-INFTY), score_combiner=ScoreMax())

    # max_step = max(map(len,vocab.keys()))
    max_step = 15
    print("K:", max_step)

    from tktkt.factories.preprocessing import RobertaPretokeniser
    return ViterbiTokeniser(RobertaPretokeniser, max_step, [objective1, objective2])


if __name__ == "__main__":
    tk = makeTokeniser()
    print(tk.prepareAndTokenise(" flatscreentelevisie"))
