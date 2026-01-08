"""
Random-access tokenisers choose their first token at any point in the string, rather than
starting at the first or last character.

Implementation adapted from my master's thesis (Bauwens, 2023). https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
Apparently this was published as "FLOTA" a year prior by Hofmann (2022). https://aclanthology.org/2022.acl-short.43.pdf
"""
from math import inf

from ...interfaces.tokenisers import *
from ...util.dicts import intersect_dicts


class LongestFirst(TokeniserWithVocabulary[WithSpecials]):
    """
    Find longest subword through the whole word.
    """

    def tokenise(self, pretoken: str) -> Tokens:
        if pretoken == "":
            return []

        # Start with biggest substring possible, then substrings 1 shorter than that, etc.
        for size in range(len(pretoken), 0, -1):
            for start in range(len(pretoken) - size + 1):
                subword = pretoken[start:start + size]
                if self.hasType(subword):
                    return self.tokenise(pretoken[:start]) + [subword] + self.tokenise(pretoken[start + size:])

        raise RuntimeError("Cannot tokenise string '", pretoken, "' because no substrings are in the vocab.")

RA_Greedy = LongestFirst
FLOTA     = LongestFirst


class HighestScoreFirst(TokeniserWithVocabulary[WithSpecials]):
    """
    Generalisation of FLOTA that associates an arbitrary score with each type in the vocabulary,
    rather than its length specifically.

    You can think of this as a greedy implementation of the ULM decoder, although calling it "greedy unigram" would be
    confusing given the naming used in the following two papers:
        https://users.ics.aalto.fi/svirpioj/online-papers/varjokallio2013asru.pdf
        https://aclanthology.org/2020.lrec-1.486.pdf
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], scores: dict[str, float], diminish_atoms: bool=False):
        """
        :param diminish_atoms: if True, the score for atoms (~characters) will be the lowest possible.
        """
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self._scores = intersect_dicts(scores, vocab)
        assert set(self._scores) == set(self.vocab), f"Missing scores for types: {list(set(self.vocab) - set(self._scores))}"

        # Set atoms to an arbitrarily low score
        if diminish_atoms:
            least_desirable_score = min(self._scores.values()) - 1
            for atom in preprocessor.getAlphabet():
                self._scores[atom] = least_desirable_score

        # Define an accelerator that gives, for each string length, the upper bound of what scores can be expected of
        # that length or below. The idea is that you consider substrings from big to small and that you can look ahead
        # at whether smaller lengths could ever give you a better result than you have found so far.
        self._max_token_length = max(map(len, self.vocab))
        self._accelerator = [-inf]
        for length in range(1,self._max_token_length+1):
            self._accelerator.append(
                max(
                    self._accelerator[-1],
                    max((score for typ, score in self._scores.items() if len(typ) == length), default=-inf)
                )
            )

    def tokenise(self, pretoken: str) -> Tokens:
        if pretoken == "":
            return []

        best_match = None
        best_score = -inf
        for size in range(min(len(pretoken),self._max_token_length), 0, -1):
            for start in range(len(pretoken) - size + 1):
                subword = pretoken[start:start+size]
                if self.hasType(subword):
                    score = self._scores[subword]
                    if score > best_score:
                        best_score = score
                        best_match = (start,size)
            if self._accelerator[size-1] < best_score:  # Best you could find at smaller sizes is worse than what you have.
                break

        if best_match is None:
            raise RuntimeError("Cannot tokenise string '", pretoken, "' because no substrings are in the vocab.")

        start, size = best_match
        return self.tokenise(pretoken[:start]) + [pretoken[start:start+size]] + self.tokenise(pretoken[start+size:])


from ..bpe.base import MergeList

class LastBPETokenFirst(HighestScoreFirst[WithSpecials]):
    """
    Find youngest BPE subword through the whole word.
    """

    def __init__(self, preprocessor: Preprocessor, bpe_vocab: Vocab[WithSpecials], bpe_merges: MergeList):
        from bpe_knockout.model.graph import MergeGraph
        scores = dict()

        graph = MergeGraph(bpe_vocab, bpe_merges)
        for m in sorted(graph.merges, reverse=True):
            scores[m.childType()] = m.priority

        assert len(list(scores.values())) == len(set(scores.values()))

        dummy_score = -1
        for t in graph.vocab:
            if graph.inAlphabet(t):
                scores[t] = dummy_score

        super().__init__(preprocessor=preprocessor, vocab=bpe_vocab, scores=scores, diminish_atoms=False)
