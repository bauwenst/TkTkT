def robbertsucks():
    from bpe_knockout.auxiliary.robbert_tokenizer import robbert_tokenizer
    s = " a flatscreen is not a (flatscreen) but it is a flat screen."
    print(s)
    print(robbert_tokenizer.tokenize(s))

    from tktkt.preparation.mappers import BYTE_TO_PSEUDO, SPACING_BYTES

    print(list(map(BYTE_TO_PSEUDO.get, SPACING_BYTES)))


def punctuation():
    from tktkt.preparation.splitters import PunctuationPretokeniser

    splitter = PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.ONLY)
    example = "energie-efficiëntie, dat is cool!"
    print(splitter.split(example))

    splitter = PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.EXCLUDED)
    print(splitter.split(example))

    splitter = PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.INCLUDED)
    print(splitter.split(example))


from tktkt.preparation.splitters import *
from tktkt.preparation.mappers import *

def steps():
    from tktkt.preparation.instances import RobertaPreprocessor, RobertaSpaceMarker, SennrichSpaceMarker, IsolatedSpaceMarker
    from tktkt.evaluation.morphological import tokeniseAsWord

    pre = PretokeniserSequence([
        PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.EXCLUDED),
        WhitespacePretokeniser(destructive=True),
        MapperAsPretokeniser(PseudoByteMapping()),
        AddSpaceMarker(SennrichSpaceMarker),
        PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.ONLY)
    ])

    word     = "ℛ𝒪𝒥ë𝒞𝒯"
    sentence = "It's a Pℛ𝒪𝒥ë𝒞𝒯, bruh! (low-key triggered)"
    clean = RobertaPreprocessor.irreversible.convert(sentence)

    print(pre.split(clean))
    quit()
    print(RobertaPreprocessor.do(word))
    print(RobertaPreprocessor.do(sentence))

    bytemapped = RobertaPreprocessor.reversible.convert(clean)
    print(clean)
    print(bytemapped)
    # TODO: The byte mapper is replacing space by Gdot and hence the space splitter can't see spaces. BAD!
    # print(RobertaPreprocessor.splitter.invertToken(sentence))

    # print(tokeniseAsWord(word, tokeniser=))


if __name__ == "__main__":
    steps()
