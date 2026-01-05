from typing import Optional

from tktkt.models.predictive.viterbi import *
from tktkt.factories.preprocessors import RobertaPreprocessor, IdentityPreprocessor
from tktkt.factories.tokenisers import Factory_BoMMaSum
from tktkt.factories.artifacts import BPE32ki_SlimPajama3M
from tktkt.interfaces.tokenisers import Preprocessor
from tktkt.interfaces.identifiers import SubwordCollection


canine_viterbi = Factory_BoMMaSum().buildTokeniser()
classifier: CharacterClassifier = canine_viterbi.objectives[0].score_generator.nested_generator.logprob_classifier

def getVocab():
    return BPE32ki_SlimPajama3M().getVocabulary()
vocab = getVocab()  # Determines how you should format the below example.
word = "Ġhorseshoe"
# word = "Ġsupercalifragilistic"


class PrototypingViterbi(ViterbiTokeniser):
    """
    Quick testing class that I can change objectives in
    to check several score grids and vocab constraints.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: SubwordCollection, max_step: Optional[int]):
        max_step = max_step or max(len(t) for t in vocab)
        generator = BoundaryPrefixAndSuffixLengthExtended(punishment=0)
        generator.setBackend(classifier)
        super().__init__(preprocessor, max_step, objectives=[
            ViterbiObjective(
                # initial_score=1,
                # score_generator=VocabularyConstraintExact(BoundaryScoresChosen(classifier, transform=DoublingMBPT()),
                #                                           vocab, reset_value=-INFTY),
                # score_combiner=ScoreProduct()
                initial_score=0,
                score_generator=generator,
                score_combiner=ScoreSum()
            ),
            ViterbiObjective(
                initial_score=0,
                score_generator=VocabularyConstraintExact(ConstantScore(), vocab, reset_value=-INFTY),
                score_combiner=ScoreSubtract()
            )
        ], degenerate=False)


def tst_verify_scoregrid():
    # Tokenise
    tk = PrototypingViterbi(IdentityPreprocessor, getVocab(), max_step=None)
    print(tk.prepareAndTokenise(word))

    # Show score grid
    constraint = tk.objectives[0].score_generator
    grid_wrapper = constraint.generateGrid(word, max_k=len(word))
    # grid_wrapper.grid = grid_wrapper.grid.astype(int)
    print(grid_wrapper)

    # Show probabilities the score grid came from
    from tst.visualisation.canine_boundaries import visualisePredictedBoundaries
    print("Probabilities:", np.exp(classifier.getPointLogProbabilities(word)).tolist())
    print("\tVisualised:", visualisePredictedBoundaries(classifier, word))

    # Show mask that turned the probabilities into the grid
    boundary_after_asmask = [1 * (np.exp(ln) > 0.5) for ln in classifier.getPointLogProbabilities(word)]
    boundary_before_asmask = [1] + boundary_after_asmask
    boundary_before_asmask[-1] = 1
    boundary_before = np.nonzero(boundary_before_asmask)
    print("Hard split indices:", boundary_before)
    print("\tbadly-zipped:", list(zip(boundary_before[:-1], boundary_before[1:])))
    print("\twell-zipped:",list(zip(boundary_before[0][:-1], boundary_before[0][1:])))


def tst_verify_that_nonboundary_does_something():
    """
    Is the score grid different for BoundaryScoresChosen vs BoundaryScoresAll?
    If yes, that explains why they look mathematically equivalent.

    Answer: They're actually functioning exactly as designed, RIP.
    """
    symmetric_transform = LinearPT(-1, +1, negate_as_complement=False)
    g1 = VocabularyConstraintExact(BoundaryScoresChosen(classifier, symmetric_transform), vocab, reset_value=-INFTY)
    g2 = VocabularyConstraintExact(BoundaryScoresAll(classifier, symmetric_transform), vocab, reset_value=-INFTY)

    with np.printoptions(suppress=True):  # https://numpy.org/devdocs/reference/generated/numpy.set_printoptions.html
        print(np.exp(classifier.getPointLogProbabilities(word)))
        print(g1.generateGrid(word, max_k=len(word)))
        print(g2.generateGrid(word, max_k=len(word)))

        print(g1.nested_generator.getBoundaryScores(word))


def tst_compare_to_robbert():
    from transformers import RobertaTokenizer

    baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    tk = PrototypingViterbi(RobertaPreprocessor, vocab, max_step=None)

    words = [" flatscreentelevisie"]
    for word in words:
        print(word)
        print("\tRobBERT BPE:",               baseline.tokenize(word))
        print("\tSame vocab, new inference:", tk.prepareAndTokenise(word))


if __name__ == "__main__":
    tst_verify_scoregrid()
    # tst_verify_that_nonboundary_does_something()
