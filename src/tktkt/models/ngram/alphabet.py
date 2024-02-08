from typing import List

from ...interfaces.general import TokeniserWithVocab


class Alphabet(TokeniserWithVocab):

    def tokenise(self, word: str) -> List[str]:
        return list(word)
