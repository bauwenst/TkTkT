from ..interfaces.tokenisers import TokeniserWithVocabulary
from ..interfaces.identifiers import AutoVocabSpecs, WithSpecials, AutoVocab


class TokeniserWithCompactedVocabDict(TokeniserWithVocabulary[WithSpecials]):
    """
    Has the same vocabulary keys as the given Tokeniser, except the IDs are remapped to be consecutive integers
    starting at 0. Helps when your vocabulary had types remove (e.g. BPE-knockout).
    """

    def __init__(self, tokeniser: TokeniserWithVocabulary, specials_specification: AutoVocabSpecs[WithSpecials], unk_type: str=None):
        compact_vocab = {k:i for i,k in enumerate(sorted(tokeniser.types(), key=tokeniser.typeToId))}
        super().__init__(preprocessor=tokeniser.preprocessor, vocab=AutoVocab.fromStrings(compact_vocab, specials_specification, unk_type))
