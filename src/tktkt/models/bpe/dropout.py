"""
Simplified version of the GuidedBPEDropout implementation, to speed it up.
"""
from typing_extensions import Self

import numpy.random as npr
from transformers import PreTrainedTokenizerFast

from ...interfaces.tokeniser import Preprocessor, Tokens
from .base import Vocab, MergeList, NonDeterministicBPETokeniser, ClassicBPE


class BPEDropout(NonDeterministicBPETokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList,
                 dropout_probability: float):
        super().__init__(
            vocab=vocab, merges=merges,

            preprocessor=preprocessor
        )
        self.p = dropout_probability
        self.rng = npr.default_rng(0)

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast, dropout_probability: float) -> Self:
        classic_implementation = ClassicBPE.fromHuggingFace(hf_bpe_tokenizer)  # Use all the logic we already have for this kind of conversion.
        return cls(
            preprocessor=classic_implementation.preprocessor,
            vocab=classic_implementation.vocab,
            merges=classic_implementation.merge_graph.getRawMerges(),

            dropout_probability=dropout_probability
        )

    def _finalTokens(self, tokens: Tokens) -> Tokens:
        buffer = " " + " ".join(tokens) + " "
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
                        if buffer[index_in_buffer:index_in_buffer+len(m[1])] == m[1]:  # Note that this is actually (believe it or not) slower than the 'in' check in the original _finalTokens.
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


class BPEDropoutNonGeometric(NonDeterministicBPETokeniser):
    """
    The outcome of which merge is selected in BPE-dropout is geometrically distributed over the IDs of the possible merges,
    sorted in order of priority. That is: the highest-priority merge has probability (1-p) of being done, the second has
    probability p(1-p), the third has p²(1-p), the fourth p³(1-p) and so on. If there are N merges, the event that no merges
    are done receives probability 1 - sum_{i=0}^{N-1} p^i(1-p).

    This class, however, gives every available merge the same probability of being chosen. If there are N possible merges,
    each is chosen with probability 1/N. (This is equivalent to classic BPE with random merge priorities.)
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList):
        super().__init__(
            vocab=vocab, merges=merges,

            preprocessor=preprocessor
        )
        self.rng = npr.default_rng(0)

    def _finalTokens(self, tokens: Tokens) -> Tokens:
        buffer = " " + " ".join(tokens) + " "
        while True:
            possible_merges = []

            types = buffer[1:-1].split(" ")
            types.pop()  # Last type won't ever start a merge, so it's useless to check merges for it.

            index_in_buffer = 0  # Start on the first space in the buffer.
            for t in types:
                # Check which merges are available.
                for m in self.merges_starting_with.get(t, []):
                    if buffer[index_in_buffer:index_in_buffer + len(m[1])] == m[1]:  # Note that this is actually (believe it or not) slower than the 'in' check in the original _finalTokens.
                        possible_merges.append((m, index_in_buffer))
                        # print("\t", m[1])
                index_in_buffer += len(t) + 1  # Move to the next space in the buffer.

            n = len(possible_merges)
            if n == 0:
                break

            random_merge, index_in_buffer = possible_merges[self.rng.integers(n)]
            buffer = buffer[:index_in_buffer] + random_merge[2] + buffer[index_in_buffer + len(random_merge[1]):]

        return buffer[1:-1].split(" ")


class BPEBreakdown(NonDeterministicBPETokeniser):
    """
    Rather than dropping merges BEFORE doing them, drop merges AFTER doing them, starting all the way at the end of the
    merge process.

    Implemented August 2024. An extension of this idea (within_tree=False) was published in the literature in June 2025,
    called "StochasTok" (https://arxiv.org/abs/2506.01687). It slightly differs in how it decides when to stop breaking
    down, and it also points out explicitly that the breakdown process is independent of BPE (already the case here).
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList,
                 within_tree: bool=False, breakdown_probability: float=0.33):
        super().__init__(
            vocab=vocab, merges=merges,

            preprocessor=preprocessor
        )
        self.rng = npr.default_rng(0)
        self.p = breakdown_probability
        self.within_tree = within_tree

    def _finalTokens(self, tokens: Tokens) -> Tokens:
        initial_tokens = list(super()._finalTokens(tokens))[::-1]  # Reverse so you can pop and append from the back.

        final_tokens = []
        while initial_tokens:
            token_to_breakdown = initial_tokens.pop()
            merges_of_this_token = self.merge_graph.merges_of[token_to_breakdown]
            if self.rng.random() < self.p and merges_of_this_token:  # High p => high entrance into this if.
                # Shortcut: if you have to stay within the merge tree, you know exactly which tokens to break into.
                if self.within_tree:
                    initial_tokens.extend(merges_of_this_token[0].parts[::-1])
                else:
                    # Otherwise: find all possible breakdowns
                    valid_splits = []
                    for i in range(len(token_to_breakdown)-1):
                        left  = token_to_breakdown[:i+1]
                        right = token_to_breakdown[i+1:]
                        if left in self.vocab and right in self.vocab:
                            valid_splits.append(i)

                    # Sample from these breakdowns
                    chosen_i = valid_splits[self.rng.integers(len(valid_splits))]

                    # Push these to the stack of tokens to be processed (left one on top of right one)
                    left  = token_to_breakdown[:chosen_i+1]
                    right = token_to_breakdown[chosen_i+1:]

                    initial_tokens.append(right)
                    initial_tokens.append(left)
            else:
                final_tokens.append(token_to_breakdown)

        return final_tokens

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast, breakdown_probability: float, within_tree: bool) -> Self:
        classic_implementation = ClassicBPE.fromHuggingFace(hf_bpe_tokenizer)  # Use all the logic we already have for this kind of conversion.
        return cls(
            preprocessor=classic_implementation.preprocessor,

            vocab=classic_implementation.vocab,
            merges=classic_implementation.merge_graph.getRawMerges(),

            breakdown_probability=breakdown_probability,
            within_tree=within_tree
        )
