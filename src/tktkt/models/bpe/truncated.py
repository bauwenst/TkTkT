"""
Limits BPE to an earlier part of training by truncating the merge list.
"""
from bpe_knockout.model.graph import MergeGraph

from .base import _DeterministicBPETokeniser, MergeList
from ...interfaces.tokenisers import *


class TruncatedBPE(_DeterministicBPETokeniser[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], merges: MergeList, delta_vocab_size: int):
        filtered_types = {m.childType() for m in sorted(MergeGraph(vocab, merges).merges)[:-delta_vocab_size]}
        # super().__init__(preprocessor=preprocessor, vocab={t:i for t,i in vocab.items() if t not in filtered_types}, merges=merges[:-delta_vocab_size])  # TODO: Possibly not compactified, but this is the same problem that happens with PickyBPE etc.
        super().__init__(preprocessor=preprocessor, vocab=vocab, merges=merges)
        for t in filtered_types:
            self.vocab.pop(t)
        self.vocab.settle()
