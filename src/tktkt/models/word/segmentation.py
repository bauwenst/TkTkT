from ...interfaces.tokenisers import *


class IdentityTokeniser(Tokeniser):
    """Returns pretokens as if they are tokens, without any further segmentation."""
    def tokenise(self, pretoken: str) -> Tokens:
        return [pretoken]

PreprocessorAsTokeniser = IdentityTokeniser
WordTokeniser           = IdentityTokeniser


class IdentityTokeniserWithVocab(TokeniserWithVocabulary[WithSpecials]):
    """Like the IdentityTokeniser, except if a pretoken is not in a given vocabulary, then it is mapped to UNK."""
    def __init__(self, preprocessor: Preprocessor, vocab: Vocab):
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self._has_unk = self.vocab.UNK is not None  # Acceleration

    def tokenise(self, pretoken: str) -> Tokens:
        return [pretoken] if self.hasType(pretoken) else [""] if self._has_unk is not None else []


class LookupTokeniser(Tokeniser):
    """
    Stores a precomputed mapping from strings to tokens.
    """

    def __init__(self, preprocessor: Preprocessor, lookup: dict[str,Tokens], do_unk: bool=True):
        super().__init__(preprocessor=preprocessor)
        self._lookup = lookup
        self._do_unk = do_unk

    def tokenise(self, pretoken: str) -> Tokens:
        return self._lookup.get(pretoken, [""] if self._do_unk else [pretoken])
