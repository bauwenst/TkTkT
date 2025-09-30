"""
Limits BPE to an earlier part of training by truncating the merge list.
"""
from .base import ClassicBPE, Vocab, MergeList, Preprocessor
from bpe_knockout.knockout.core import MergeGraph


class TruncatedBPE(ClassicBPE):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList, delta_vocab_size: int):
        filtered_types = {m.childType() for m in sorted(MergeGraph(vocab, merges).merges)[:-delta_vocab_size]}
        super().__init__(preprocessor=preprocessor, vocab={t:i for t,i in vocab.items() if t not in filtered_types}, merges=merges[:-delta_vocab_size])  # TODO: Possibly not compactified, but this is the same problem that happens with PickyBPE etc.
