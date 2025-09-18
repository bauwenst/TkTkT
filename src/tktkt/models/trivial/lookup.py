from ...interfaces.tokeniser import Tokeniser, Tokens, Preprocessor


class SegmentationLookup(Tokeniser):
    """
    Stores a precomputed mapping from strings to tokens.
    """

    def __init__(self, preprocessor: Preprocessor, lookup: dict[str,Tokens], unk_type: str=None):
        super().__init__(preprocessor=preprocessor)
        self._lookup = lookup
        self._unk = unk_type
        self._has_unk = self._unk is not None

    def tokenise(self, pretoken: str) -> Tokens:
        return self._lookup.get(pretoken, [self._unk] if self._has_unk else [pretoken])
