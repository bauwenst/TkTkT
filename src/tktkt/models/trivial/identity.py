from typing import List

from ...interfaces.tokeniser import Tokeniser, TokeniserWithVocabDict


class IdentityTokeniser(Tokeniser):
    """Returns pretokens as if they are tokens, without any further segmentation."""
    def tokenise(self, pretoken: str) -> List[str]:
        return [pretoken]

PreprocessorAsTokeniser = IdentityTokeniser


class IdentityTokeniserWithVocab(TokeniserWithVocabDict):
    """Like the IdentityTokeniser, except if a pretoken is not in a given vocabulary, then it is mapped to UNK."""
    def tokenise(self, pretoken: str) -> List[str]:
        return [pretoken] if self.hasType(pretoken) else [self.unk] if self.unk else []
