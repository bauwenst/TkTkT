from ..interfaces.tokeniser import TokeniserWithFiniteTypeDomain, TokeniserWithVocabDict


class TokeniserWithCompactedVocabDict(TokeniserWithVocabDict):
    """
    Has the same vocabulary keys as the given Tokeniser, except the IDs are remapped to be consecutive integers
    starting at 0. Helps when your vocabulary had types remove (e.g. BPE-knockout).
    """

    def __init__(self, tokeniser: TokeniserWithFiniteTypeDomain, unk_type: str=None):
        compact_vocab = {k:i for i,k in enumerate(sorted(tokeniser.types(), key=tokeniser.typeToId))}
        super().__init__(tokeniser.preprocessor, compact_vocab, unk_type=unk_type)
