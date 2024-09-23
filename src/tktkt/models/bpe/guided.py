from typing import List, Iterable, MutableSequence, Union, Optional
from typing_extensions import Self

import numpy.random as npr
from math import exp, log, inf

from transformers import PreTrainedTokenizerFast

from ...preparation.boundaries import BoundaryMarker
from ...interfaces.tokeniser import Preprocessor
from ..viterbi.objectives_guided import CharacterClassifier
from .base import Vocab, MergeList, NonDeterministicBPETokeniser, ClassicBPE


class ConstantCharacterClassifier(CharacterClassifier):

    def __init__(self, p: float):
        self.logp = log(p) if p > 0.0 else -inf

    def getPointLogProbabilities(self, pretoken: str) -> MutableSequence[float]:
        return [self.logp for _ in range(len(pretoken))]


class GuidedBPEDropout(NonDeterministicBPETokeniser):
    """
    Generalisation of BPE-dropout which varies the probability of every merge being applied in the given string based on
    a classifier that predicts whether the split that would be removed by the merge should be kept (e.g. because it is
    a morpheme boundary).

    If you want to use vanilla BPE-dropout, you should probably use a more efficient version than this.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList, boundary_marker: BoundaryMarker,
                 dropout_probability: Union[float,CharacterClassifier], always_dropout_above: Optional[float]=None, unk_type: str=None):
        """
        TODO: To improve deterministic performance, you might want to use a custom threshold. For example, if CANINE thinks
              a boundary should be there with 30% probability, perhaps that's actually enough.
        TODO: To improve non-deterministic performance, you might want to apply companding
              (e.g. p^{1/a} or (1-e^{-bx})/(1-e^{-b}) or log(1+cx)/log(1+c) or a sigmoid, see https://www.desmos.com/calculator/drttocdsqx)
              to the dropout probabilities (essentially, you make the model more certain about any sort of signal it thinks is there for a boundary).

        :param dropout_probability: Model that generates the dropout probability per position of a string to be segmented.
                                    If a floating point number, this will be repeated for all strings for all positions.
        :param always_dropout_above: In not None, pull the dropout probability to 1 if it is >= this, and pull it to 0
                                     otherwise. This makes dropout deterministic (the same boundary will always either
                                     drop or not drop). If you want to experiment with this, a sane value is 0.5.
                                     |
                                     Obviously has little use when using a constant dropout probability, because then
                                     you always drop out (dropout_probability >= hard_boundary_above) or never drop out.
        """
        super().__init__(
            # Init
            vocab=vocab, merges=merges,
            unk_type=unk_type,

            # Prep
            preprocessor=preprocessor,
            boundary_marker=boundary_marker
        )

        self.classifier = dropout_probability if not isinstance(dropout_probability, float) else ConstantCharacterClassifier(dropout_probability)
        self.deterministic           = always_dropout_above is not None
        self.deterministic_threshold = always_dropout_above
        self.rng = npr.default_rng(0)

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast,
                        dropout_probability: Union[float,CharacterClassifier], always_dropout_above: Optional[float]=None) -> Self:
        classic_implementation = ClassicBPE.fromHuggingFace(hf_bpe_tokenizer)  # Use all the logic we already have for this kind of conversion.
        return cls(
            preprocessor=classic_implementation.preprocessor,
            vocab=classic_implementation.vocab,
            merges=classic_implementation.merge_graph.getRawMerges(),
            boundary_marker=classic_implementation._boundary_marker,
            unk_type=classic_implementation.unk,

            dropout_probability=dropout_probability,
            always_dropout_above=always_dropout_above
        )

    def applyMerges(self, sequence_of_nonspaces: Iterable[str]) -> List[str]:
        """
        Implementation where each merge instance is treated separately, and the index of each space is also tracked to
        be able to verify if it is allowed to disappear.
        """
        p_forbidden_to_concatenate = [exp(L) for L in self.classifier.getPointLogProbabilities("".join(sequence_of_nonspaces))]

        buffer = " " + " ".join(sequence_of_nonspaces) + " "
        while True:
            # print(buffer)
            possible_merges = []

            tokens = buffer[1:-1].split(" ")
            tokens.pop()

            index_in_lookup = -1  # Start before the first character. When the first token has length 3, e.g. "abc", you have to look up the probability at index 2.
            index_in_buffer = 0  # Start on the first space in the buffer.
            for t in tokens:
                # Check if a merge is allowed after this type. If not, don't bother checking for possible merges.
                # TODO: This only works for a BPE tokeniser with binary merges, because for n-ary merges, you should check n-1
                #       spaces, not just the space after the first token.
                index_in_lookup += len(t)
                rng_result = self.deterministic_threshold if self.deterministic else self.rng.random()  # Sneaky trick: instead of thresholding all the probabilities to 0/1 and then comparing that to a random number (where clearly the random number's value never matters), we compare with a deterministic number instead. All probabilities under 0.5 won't drop out (== True `if` statement) and all above 0.5 will.
                if rng_result > p_forbidden_to_concatenate[index_in_lookup]:  # Sanity check: when p_dropout = 0.9, you expect 90% of all RNGs to skip the rest of the loop (only 10% make it above p). If you clip p to 0/1, you would expect 100% dropout, and indeed, 0.5 never lies above 0.9.
                    # Check which merges are available.
                    for m in self.merges_starting_with.get(t, []):
                        # print(t, m, "compared to", buffer[index_in_buffer:index_in_buffer+len(m[1])])
                        if buffer[index_in_buffer:index_in_buffer+len(m[1])] == m[1]:
                            possible_merges.append((m,index_in_buffer))
                            # print("\t", m[1])
                # else:
                #     print("Skipped merging with type", t, f"because {rng_result} <= P[boundary at {index_in_lookup}] ==", p_forbidden_to_concatenate[index_in_lookup])
                index_in_buffer += len(t) + 1

            # print(possible_merges)
            if not possible_merges:
                break

            # BPE-dropout executes exactly one merge instance per iteration, whilst BPE executes all instances of the same merge at once.
            best_merge, index_in_buffer = min(possible_merges)
            buffer = buffer[:index_in_buffer] + best_merge[2] + buffer[index_in_buffer+len(best_merge[1]):]
            # print("\t", best_merge)

        return buffer[1:-1].split(" ")

    def getName(self):
        if self.deterministic:
            return super().getName() + f"(deterministic, 𝜃={self.deterministic_threshold})"
        else:
            return super().getName() + "(non-deterministic)"