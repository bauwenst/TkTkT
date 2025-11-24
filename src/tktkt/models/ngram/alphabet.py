from typing import List, Mapping, Iterable

from ...interfaces.tokeniser import TokeniserWithFiniteIdRange


UNICODE_VALUES = 149_813 + 65

class OrdMapping(Mapping):
    """
    Mapping representing the ord() function.
    """

    def get(self, key):
        return ord(key)

    def keys(self):
        raise RuntimeError("ord() mapping has an indefinite key domain.")

    def values(self):
        return range(0, UNICODE_VALUES)

    def items(self):
        raise RuntimeError("ord() mapping has an indefinite key domain.")


class UnicodeTokeniser(TokeniserWithFiniteIdRange):

    def ids(self) -> Iterable[int]:
        return range(UNICODE_VALUES)

    def hasId(self, i: int) -> bool:
        return i in self.ids()

    def tokenise(self, word: str) -> List[str]:
        return list(word)

    def typeToId(self, t: str) -> int:
        return ord(t)

    def idToType(self, i: int) -> str:
        return chr(i)

    def getVocabSize(self) -> int:
        """
        149 813 codepoints and 65 control characters.
        https://www.unicode.org/versions/stats/charcountv15_1.html
        """
        return UNICODE_VALUES
