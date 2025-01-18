from typing import List

from ...interfaces.tokeniser import Tokeniser, TokeniserWithVocabDict


class IdentityTokeniser(Tokeniser):
    def tokenise(self, pretoken: str) -> List[str]:
        return [pretoken]


class IdentityTokeniserWithVocab(TokeniserWithVocabDict):
    def tokenise(self, pretoken: str) -> List[str]:
        if self.hasType(pretoken):
            return [pretoken]
        elif self.unk:
            return [self.unk]
        else:
            return []
    