from typing import List
from collections import Counter

from .segmentation import BpeTokeniser, WordSplitter, Vocab


class TrimmedBPETokeniser(BpeTokeniser):
    """
    Implementation of Yuval Pinter's TrimmedBPE tokeniser, which first applies BPE like normal, and then
    recursively undoes applied merges until all tokens are NOT in a predefined set of illegal types.
    """

    def __init__(self, pretokeniser: WordSplitter, vocab: Vocab, merges: List[str]):
        super().__init__(pretokeniser, vocab, merges)
        self.disabled = set()

    def tokenise(self, pretoken: str) -> List[str]:
        # Do BPE
        old_tokens = super().tokenise(pretoken)

        # Undo BPE
        new_tokens = []
        for token in old_tokens:
            new_tokens.extend(self.recursivelyDecompose(token))
        return new_tokens

    def recursivelyDecompose(self, token: str) -> List[str]:
        if token not in self.vocab:
            raise ValueError(f"Cannot decompose token that doesn't have a type in the vocabulary: {token}")

        if token not in self.disabled:
            return [token]
        else:
            part1, part2 = self.core.merge_graph.merges_of[token][0].parts
            return self.recursivelyDecompose(part1) + self.recursivelyDecompose(part2)

    def trim(self, corpus: Counter, threshold: int=0):
        """
        Disable all types that appear <= threshold times when tokenising the given corpus.
        This process is done in parallel (find all types, then disable them) rather than serially (find a type, disable
        it, find a type with the updated tokeniser, disable it, ...).

        I suspect that this method is idempotent, but I'm too lazy to prove it.
        """
        # Get counts
        type_counts = Counter()
        for word, count in corpus.items():
            for token in self.prepareAndTokenise(word):
                type_counts[token] += count

        # Find and disable infrequent types
        for t in self.vocab:
            if type_counts[t] <= threshold:
                try:
                    self._disableType(t)
                except:  # t is part of the alphabet.
                    pass

    def _disableType(self, type_to_disable: str):
        if type_to_disable not in self.vocab or not self.core.merge_graph.merges_of[type_to_disable]:
            raise ValueError(f"Cannot trim a type from the vocabulary that cannot be decomposed further: {type_to_disable}")

        self.disabled.add(type_to_disable)
