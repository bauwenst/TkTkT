from tst.preamble import *

from tktkt.evaluation.entropy import getTokenDistributionFromSentences_and_analyse
from tktkt.models.identity.segmentation import IdentityTokeniserWithVocab
from tktkt.factories.preprocessing import Preprocessor, WhitespacePretokeniser
from tktkt.util.types import NamedIterable


def toy():
    corpus = NamedIterable([
        "a a a b b c c a a",
        "c c c c"
    ], "dummy")
    tk = IdentityTokeniserWithVocab(Preprocessor(splitter=WhitespacePretokeniser()), {"a": 0, "b": 1, "c": 2, "d": 3})

    print(getTokenDistributionFromSentences_and_analyse(
        tk, corpus,
        window_size=5, stride=2
    ))


if __name__ == "__main__":
    toy()
