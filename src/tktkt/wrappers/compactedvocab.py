from ..interfaces.tokeniser import TokeniserWithVocabDict


class TokeniserWithCompactedVocabDict(TokeniserWithVocabDict):
    """
    Has the same vocabulary keys as the given Tokeniser, except the IDs are remapped to be consecutive integers
    starting at 0. Helps when your vocabulary had types remove (e.g. BPE-knockout).
    """

    def __init__(self, tokeniser: TokeniserWithVocabDict):
        compact_vocab = {k:i for i,k in enumerate(sorted(tokeniser.vocab.keys(), key=tokeniser.vocab.get))}
        super().__init__(tokeniser.preprocessor, compact_vocab, unk_type=tokeniser.unk)
