"""
Tests punctuation pretokenisation to support Text2PhonemeSequence (T2PS, pronounced TeetwoPiece).
"""
from tst.preamble import *

from tktkt.preparation.splitters import *
from tktkt.preparation.mappers import *

phoneme_pretokeniser = PretokeniserSequence([
    OnWhitespace(),
    IsolatePunctuation(
        hyphen_mode=HyphenMode.INCLUDED,
        protect_apostrophes_without_spaces=True
    )
])


def a():
    map_seq = MapperSequence([
        DilatePretokens(phoneme_pretokeniser),
        AsPhonemes()
    ])

    s = "My friend, I can't believe you've done this. Have you heard of 'vectors' or not?"
    print(phoneme_pretokeniser.split(s))
    print(map_seq.convert(s))


def b():
    split_seq = PretokeniserSequence([
        phoneme_pretokeniser,
        EnglishApostrophes(do_nt=True)
    ])

    sentences = [
        "I don't think you have the facilities for that; I've checked, although I'm quite sure I won't or shouldn't've done that.",
        "Don't you know you should've drawn a 'vector' there? I've"
    ]

    for s in sentences:
        print("/".join(phoneme_pretokeniser.split(s)))
        print("/".join(split_seq.split(s)))


if __name__ == "__main__":
    a()
    # b()