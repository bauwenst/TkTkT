"""
Random-access tokenisers choose their first token at any point in the string, rather than
starting at the first or last character.

Implementation adapted from my master's thesis (Bauwens, 2023). https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
Apparently this was published a year prior by Hofmann (2022). https://aclanthology.org/2022.acl-short.43.pdf
"""
from typing import List

from ...interfaces.general import TokeniserWithVocab, Vocab
from ...preparation.spacemarking import SpaceMarker
from ...preparation.splitters import WordSplitter


class RA_Greedy(TokeniserWithVocab):

    def __init__(self, vocab: Vocab, marker: SpaceMarker):
        super().__init__(pretokeniser=WordSplitter(marker))
        self.vocab = vocab

    def tokenise(self, word: str) -> List[str]:
        if word == "":
            return []

        # Find longest subword through the whole word.
        # --- New implementation: for each length, check each start, and terminate immediately on the first match ---
        found = False
        for size in range(len(word), 0, -1):
            for start in range(len(word) - size + 1):
                subword = word[start:start + size]
                # if self.vocab.hasWord(subword):
                if subword in self.vocab:
                    found = True
                    break
            if found:
                break

        if not found:
            raise RuntimeError("Cannot tokenise string '", word, "' because no substrings are in the vocab.")

        return self.tokenise(word[:start]) + \
            [subword] + \
            self.tokenise(word[start + size:])