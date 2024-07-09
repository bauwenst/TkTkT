"""
Simplified version of the GuidedBPEDropout implementation, to speed it up.

FIXME: Because the BTE class has caching enabled on .tokenise, it's quite possible that dropout actually doesn't work
       (since the same string will default to the same tokenisation).
"""
from typing import List, Iterable

import numpy.random as npr
from transformers import PreTrainedTokenizerFast

from ...preparation.boundaries import BoundaryMarker
from ...interfaces.tokeniser import Preprocessor
from .base import Vocab, MergeList, ClassicBPE


class BPEDropout(ClassicBPE):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList, boundary_marker: BoundaryMarker,
                 dropout_probability: float, unk_type: str=None):
        super().__init__(
            vocab=vocab, merges=merges,
            unk_type=unk_type,

            preprocessor=preprocessor,
            boundary_marker=boundary_marker,
        )
        self.p = dropout_probability
        self.rng = npr.default_rng(0)

    @staticmethod
    def fromHuggingFace(hf_bpe_tokenizer: PreTrainedTokenizerFast, dropout_probability: float) -> "BPEDropout":
        classic_implementation = super().fromHuggingFace(hf_bpe_tokenizer)  # Use all the logic we already have for this kind of conversion.
        return BPEDropout(
            preprocessor=classic_implementation.preprocessor,
            vocab=classic_implementation.vocab,
            merges=classic_implementation.merge_graph.getRawMerges(),
            boundary_marker=classic_implementation.boundary_marker,
            unk_type=classic_implementation.unk,

            dropout_probability=dropout_probability
        )

    def applyMerges(self, sequence_of_nonspaces: Iterable[str]) -> List[str]:
        buffer = " " + " ".join(sequence_of_nonspaces) + " "
        while True:
            possible_merges = []

            types = buffer[1:-1].split(" ")
            types.pop()  # Last type won't ever start a merge, so it's useless to check merges for it.

            index_in_buffer = 0  # Start on the first space in the buffer.
            for t in types:
                # Check if a merge is allowed after this type. If not, don't bother checking for possible merges. TODO: Not sure if this is still correct in BTE when you have a sequence t1 t2 t3 and there exists both a merge t1+t2 as well as t1+t2+t3 (which would necessarily have higher priority than t1+t2).
                if self.rng.random() > self.p:  # Sanity check: when p = 0.9, you expect 90% of all RNGs to not reach above p and hence skip the rest of the loop.
                    # Check which merges are available.
                    for m in self.merges_starting_with.get(t, []):
                        if buffer[index_in_buffer:index_in_buffer+len(m[1])] == m[1]:  # Note that this is actually (believe it or not) slower than the 'in' check in the original applyMerges.
                            possible_merges.append((m,index_in_buffer))
                            # print("\t", m[1])
                index_in_buffer += len(t) + 1  # Move to the next space in the buffer.

            if not possible_merges:
                break

            # In BPE-dropout, a high-priority merge that is excluded one iteration can still be done the next iteration.
            # The alternative, where surviving instances of the same merge are all done simultaneously (i.e. more than
            # one of the elements in the possible_merges list is executed) and dead instances are perhaps forgotten
            # forever (i.e. remembering a history of excluded merges), would require much more than a simple min().
            best_merge, index_in_buffer = min(possible_merges)

            # BPE-dropout does exactly one merge instance per iteration, whilst BPE does all instances of
            # the same merge at once. Hence why BPE can use .replace() whilst here you need to cut out exactly one space.
            # buffer = buffer.replace(best_merge[1], best_merge[2])
            buffer = buffer[:index_in_buffer] + best_merge[2] + buffer[index_in_buffer+len(best_merge[1]):]

        return buffer[1:-1].split(" ")
