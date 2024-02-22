from typing import List

from ..preparation.splitters import Pretokeniser, IdentityPretokeniser
from ..preparation.mappers import TextMapper, InvertibleTextMapper, IdentityMapper


class Preprocessor:
    """
    Applies the following transformations to a string of text:

        raw text
          v  one-time mapping
        clean text                       clean text
          v  invertible mapping               ^
        altered text                     altered text
          v  splitter                         ^
        pretokens   --- (tokeniser) ---->   tokens
    """

    def __init__(self, uninvertible_mapping: TextMapper=None, invertible_mapping: InvertibleTextMapper=None, splitter: Pretokeniser=None):
        self.irreversible = uninvertible_mapping or IdentityMapper()
        self.reversible   = invertible_mapping   or IdentityMapper()
        self.splitter     = splitter             or IdentityPretokeniser()

    def do(self, text: str) -> List[str]:
        return self.splitter.split(self.reversible.convert(self.irreversible.convert(text)))

    def undo(self, tokens: List[str]) -> str:
        return self.reversible.invert(self.splitter.unsplit(tokens))

    def undo_per_token(self, tokens: List[str]) -> List[str]:
        return [self.reversible.invert(self.splitter.invertToken(token)) for token in tokens]