from tst.preamble import *

from tktkt.evaluation.entropy import getTokenDistributionFromSentences_and_analyse
from tktkt.models.trivial.identity import IdentityTokeniserWithVocab
from tktkt.factories.preprocessing import Preprocessor, WhitespacePretokeniser
from tktkt.util.types import NamedIterable


def toy():
    corpus = NamedIterable([
        "a a a b b c c a a",
        "c c c c"
    ], "dummy").tqdm()  # Windows: [a,a,a,b,b] 2/5 [a,b,b,c,c] 3/5 [b,c,c,a,a] 3/5 [c,a,a,c,c] 2/5 [a,c,c,c,c] 2/5
    tk = IdentityTokeniserWithVocab(Preprocessor(splitter=WhitespacePretokeniser()), {"a": 0, "b": 1, "c": 2, "d": 3})

    distribution, stats = getTokenDistributionFromSentences_and_analyse(
        tk, corpus,
        window_size=5, stride=2
    )
    print(distribution)
    print(stats)  # MATTR should be exactly 48%.


if __name__ == "__main__":
    toy()
