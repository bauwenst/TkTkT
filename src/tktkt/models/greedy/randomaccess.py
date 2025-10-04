"""
Random-access tokenisers choose their first token at any point in the string, rather than
starting at the first or last character.

Implementation adapted from my master's thesis (Bauwens, 2023). https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
Apparently this was published as "FLOTA" a year prior by Hofmann (2022). https://aclanthology.org/2022.acl-short.43.pdf
"""
from typing import List
from math import inf

from ...interfaces import Preprocessor
from ...interfaces.tokeniser import TokeniserWithVocabDict, Vocab
from ...util.types import Tokens


class RA_Greedy(TokeniserWithVocabDict):
    """
    Find longest subword through the whole word.
    """

    def tokenise(self, pretoken: str) -> List[str]:
        if pretoken == "":
            return []

        # Start with biggest substring possible, then substrings 1 shorter than that, etc.
        for size in range(len(pretoken), 0, -1):
            for start in range(len(pretoken) - size + 1):
                subword = pretoken[start:start + size]
                if self.hasType(subword):
                    return self.tokenise(pretoken[:start]) + [subword] + self.tokenise(pretoken[start + size:])

        raise RuntimeError("Cannot tokenise string '", pretoken, "' because no substrings are in the vocab.")


FLOTA = RA_Greedy


from bpe_knockout.knockout.core import MergeGraph
from ..bpe.base import MergeList

class LastBPETokenFirst(TokeniserWithVocabDict):
    """
    Find youngest BPE subword through the whole word.

    TODO: You can speed this up a little bit using a data structure that stores, for each length in the vocabulary,
          the highest merge priority for that length (max'ed with lower lengths). That is: when you have seen all
          substrings of length L and your best match is currently of priority P, you know you don't have to go to L-1...1.
    """

    def __init__(self, preprocessor: Preprocessor, bpe_vocab: Vocab, bpe_merges: MergeList):
        super().__init__(preprocessor=preprocessor, vocab=bpe_vocab)
        self._scores: dict[str,int] = dict()

        graph = MergeGraph(bpe_vocab, bpe_merges)
        for m in sorted(graph.merges, reverse=True):
            self._scores[m.childType()] = m.priority

        assert len(list(self._scores.values())) == len(set(self._scores.values()))

        dummy_score = max(self._scores.values()) + 1
        for t in graph.vocab:
            if graph.inAlphabet(t):
                self._scores[t] = dummy_score

        assert len(self._scores) == len(self.vocab)

    def tokenise(self, pretoken: str) -> List[str]:
        if pretoken == "":
            return []

        best_match = None
        best_score = -inf
        for size in range(len(pretoken), 0, -1):
            for start in range(len(pretoken) - size + 1):
                subword = pretoken[start:start+size]
                if self.hasType(subword):
                    score = self._scores[subword]
                    if score > best_score:
                        best_score = score
                        best_match = (start,size)

        if best_match is None:
            raise RuntimeError("Cannot tokenise string '", pretoken, "' because no substrings are in the vocab.")

        start, size = best_match
        return self.tokenise(pretoken[:start]) + [pretoken[start:start+size]] + self.tokenise(pretoken[start+size:])
