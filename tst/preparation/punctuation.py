from tst.preamble import *

from tktkt.factories.preprocessors import *


def tst_distinctPunctuation():
    p = PretokeniserSequence([
        IsolatePunctuation(HyphenMode.INCLUDED),
        GroupDistinctPunctuation()
    ])

    print(p.split("""This is an "example": we want to separate unequal punctuation...! Anyway, we have 'tokens', etc. :-)"""))


def tst_reverse():
    p = PretokeniserSequence([
        OnWhitespace(),
        IsolatePunctuation(HyphenMode.INCLUDED),
        InsertReverse(),
        AddWordBoundary(RobertaSpaceMarker)
    ])
    print(p.split(" This is an example sentence of reversing tokens."))


if __name__ == "__main__":
    tst_reverse()
