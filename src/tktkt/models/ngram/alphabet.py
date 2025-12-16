from typing import List, Iterable

from ...interfaces import Preprocessor, Vocab
from ...interfaces.identifiers import WithSpecials
from ...interfaces.tokenisers import TokeniserWithVocabulary


UNICODE_VALUES = 149_813 + 65


class UnicodeTokeniser(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, specials: WithSpecials):
        super().__init__(preprocessor=preprocessor, vocab=Vocab([], specials=specials, unk_id=None))

    def tokenise(self, word: str) -> List[str]:
        return list(word)

    # Iterate
    def types(self) -> Iterable[str]:
        return map(self.idToType, self.ids())

    def ids(self) -> Iterable[int]:
        return range(UNICODE_VALUES)

    # Membership
    def hasId(self, i: int) -> bool:
        return i in self.ids()

    def hasType(self, t: str) -> bool:
        return self.hasId(self.typeToId(t))

    # Convert
    def typeToId(self, t: str) -> int:
        return ord(t)

    def idToType(self, i: int) -> str:
        return chr(i)

    # def getVocabSize(self) -> int:
    #     """
    #     149 813 codepoints and 65 control characters.
    #     https://www.unicode.org/versions/stats/charcountv15_1.html
    #     """
    #     return UNICODE_VALUES
