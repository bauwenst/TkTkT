from typing import List

from ...interfaces.tokeniser import TokeniserWithVocab


class UnicodeTokeniser(TokeniserWithVocab):

    def tokenise(self, word: str) -> List[str]:
        return list(word)

    def getVocabSize(self) -> int:
        """
        149 813 codepoints and 65 control characters.
        https://www.unicode.org/versions/stats/charcountv15_1.html
        """
        return 149_878