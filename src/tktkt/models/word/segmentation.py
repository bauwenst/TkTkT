from typing import List

from ...interfaces.tokeniser import Tokeniser, TokeniserWithVocabDict, Tokens, Preprocessor


class IdentityTokeniser(Tokeniser):
    """Returns pretokens as if they are tokens, without any further segmentation."""
    def tokenise(self, pretoken: str) -> List[str]:
        return [pretoken]

PreprocessorAsTokeniser = IdentityTokeniser
WordTokeniser           = IdentityTokeniser


class IdentityTokeniserWithVocab(TokeniserWithVocabDict):
    """Like the IdentityTokeniser, except if a pretoken is not in a given vocabulary, then it is mapped to UNK."""
    def tokenise(self, pretoken: str) -> List[str]:
        return [pretoken] if self.hasType(pretoken) else [self.unk] if self.unk else []


class LookupTokeniser(Tokeniser):
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
