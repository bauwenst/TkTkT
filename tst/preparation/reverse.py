"""
Rather than give a model the result of applying simple preprocessing + tokenisation,
insert the reverse of each pretoken before tokenising and see if the model can figure this out.
"""
from tktkt.preparation.mappers import PseudoByteMapping
from tktkt.preparation.splitters import *


class ModernEnglishPreprocessorWithReverse(PretokeniserSequence):
    def __init__(self, marker: BoundaryMarker, do_reverse: bool):
        super().__init__(
            [
                PunctuationPretokeniser(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
                WhitespacePretokeniser(destructive=True)
            ] +\
            do_reverse * [
                InsertReverse(),  # Insert the reverse BEFORE byte mapping and BEFORE the word boundary, because the reverse of byte mappings doesn't exist and there are no types in the vocab with the word boundary on the opposite end.
            ] +\
            [
                MapperAsPretokeniser(PseudoByteMapping()),
                EnglishApostrophes(do_nt=True),
                AddWordBoundary(marker),
                IsolateDigits(),
                PunctuationPretokeniser(HyphenMode.ONLY),
            ]
        )


def tst_reversingprep():
    """
    A test to see what the model would roughly see by using the reversing preprocessor.

    Hmmmm, kind of fragmented, and also some other preprocessor stuff causes some words to correspond to more than
    one pretoken.
    """
    from transformers import AutoTokenizer

    from tktkt.interfaces import Preprocessor
    from tktkt.preparation.instances import RobertaSpaceMarker
    from tktkt.preparation.huggingface import HuggingFaceNormaliser, tn
    from tktkt.preparation.mappers import IdentityMapper
    from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser

    # Define preprocessor
    reversing_pretokeniser = ModernEnglishPreprocessorWithReverse(RobertaSpaceMarker, do_reverse=True)
    preprocessor = Preprocessor(
        uninvertible_mapping=HuggingFaceNormaliser(tn.NFKC()),
        invertible_mapping=IdentityMapper(),
        splitter=reversing_pretokeniser
    )

    # Test preprocessor
    s = "This is a sentence like any other sentence brother (but it has 69420 \"strange\" quirks). That's it."
    print(preprocessor.do(s))

    # Define tokeniser
    # tk = Builder_English_BPE_native().buildTokeniser()
    tk = HuggingFaceTokeniser(wrapped_tokeniser=AutoTokenizer.from_pretrained("roberta-base"))
    tk.preprocessor = preprocessor

    # Test preprocessor + tokeniser
    print(tk.prepareAndTokenise(s))

    # ---
    pretokens = preprocessor.do(s)
    for forward, backward in zip(pretokens[::2], pretokens[1::2]):
        print(forward, backward)
        print(tk.tokenise(forward), tk.tokenise(backward))


if __name__ == "__main__":
    tst_reversingprep()
