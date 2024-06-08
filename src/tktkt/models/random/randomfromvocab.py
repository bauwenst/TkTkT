from typing import List, Iterable

import numpy.random as npr
import itertools
from tqdm.auto import tqdm

from tktkt.util.functions import relu

from ...util.printing import intsep
from ...interfaces.tokeniser import TokeniserWithVocabDict, Preprocessor, Vocab


class RandomSegmentationFromVocab(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None):
        super().__init__(preprocessor, vocab, unk_type)
        self.rng = npr.default_rng(0)

    def tokenise(self, pretoken: str) -> List[str]:
        """
        Get a random segmentation that is possible using the tokeniser's vocabulary.
        """
        segmentations = generateSegmentationIndices_exponentialSpace(pretoken, self.vocab)
        return segmentUsingIndices(pretoken, segmentations[self.rng.choice(len(segmentations))])


def generateSegmentationIndices_exponentialSpace(text: str, vocab: Vocab) -> List[List[int]]:
    """
    We have a function possibleSegmentations() somewhere that gives the AMOUNT of possible segmentations, but not
    what they are. A very similar Viterbi algorithm works here, except the history you store is not an integer but
    a collection of lists (where the amount computed at each step of the above algorithm is the size of that collection).

    A segmentation is represented by the list of start indices of its tokens. Since we actually need to return every
    segmentation and in a Viterbi algorithm you finish all segmentations in parallel instead of sequentially, this
    algorithm takes O(2^n) space in the worst case, and to construct that, you also need O(2^n) time. However, if you
    are lucky, space and time stay lower than exponential.
    """
    # TODO: What should happen is that you should have a beam size of the maximal amount of segmentations you can tolerate,
    #       and once you reach that in the trellis, you don't create new lists, only extend old ones, and warn that you will
    #       receive an approximation.
    if len(text) > 27:
        raise RuntimeWarning(f"You better not generate all segmentations of the string '{text}' (length {len(text)}): that's up to {intsep(2**(len(text)-1))} lists!")

    unique_segmentations_up_to: List[List[List[int]]] = [[] for _ in range(len(text)+1)]  # Each element is a list of paths to get from "" to text[:index] (exclusive indexing).
    unique_segmentations_up_to[0].append([])
    for from_this_char in range(len(text)):  # The last index you can start from is the last character, len(pretoken)-1.
        for to_this_char in range(from_this_char+1, len(text)+1):  # The last index is len(pretoken), the exclusive upper bound for the entire string.
            if text[from_this_char:to_this_char] in vocab:
                # For each segmentation that exists to the from_char, a new segmentation exists to the to_char now.
                for seg in unique_segmentations_up_to[from_this_char]:
                    unique_segmentations_up_to[to_this_char].append(seg + [from_this_char])

    return [seg for seg in unique_segmentations_up_to[-1]]


def segmentUsingIndices(text: str, starts_of_tokens: List[int]) -> List[str]:
    return [text[start_idx:end_idx] for start_idx, end_idx in zip(starts_of_tokens, starts_of_tokens[1:] + [len(text)])]


def generateSegmentations(text: str, vocab: Vocab) -> List[List[str]]:
    return [segmentUsingIndices(text, indices)
            for indices in generateSegmentationIndices_exponentialSpace(text, vocab)]


def generateSegmentationIndices_exponentialTime(text: str, vocab: Vocab) -> Iterable[List[int]]:
    """
    Naive implementation that just checks every possible one of the 2^{N-1} possible segmentations and outputs
    the valid ones. It has guaranteed O(2^n) time complexity, but it also has O(1) space complexity.

    Note that Python does about 10 000 000 loop iterations per second. That means for a string of length 28, which has
    2^27 possible segmentations, you need a little over 10 seconds (times some factor).
    """
    for n_splits in range(len(text)):
        for c in itertools.combinations(range(1,len(text)), r=n_splits):
            indices = [0] + list(c)
            tokens = segmentUsingIndices(text, indices)
            if all(t in vocab for t in tokens):
                yield indices


def generateSegmentationIndices_fixedSpace(text: str, vocab: Vocab, max_prefix_length: int=22) -> Iterable[List[int]]:
    """
    Hybrid approach: use Viterbi to reduce a bunch of segmentation possibilities, store the results in lists, and use
    them as starting points for generating segmentations on-the-fly.

    :param max_prefix_length: Length of the string to generate all possible segmentations for in parallel, held in
                              memory all at once.
    """
    original_text = text
    remaining = relu(len(original_text) - max_prefix_length)
    # partial_segmentations = generateSegmentationIndices_exponentialSpace(text[:prefix_length], vocab)  # Doesn't work! You need to check segmentations that end anywhere, not just at the end of the partial string.

    # Part 1: Quadratic
    text = original_text[:max_prefix_length]

    unique_segmentations_up_to: List[List[List[int]]] = [[] for _ in range(len(text)+1)]  # Each element is a list of paths to get from "" to text[:index] (exclusive indexing).
    unique_segmentations_up_to[0].append([])
    for from_this_char in range(len(text)):  # The last index you can start from is the last character, len(pretoken)-1.
        for to_this_char in range(from_this_char+1, len(text)+1):  # The last index is len(pretoken), the exclusive upper bound for the entire string.
            if text[from_this_char:to_this_char] in vocab:
                # For each segmentation that exists to the from_char, a new segmentation exists to the to_char now.
                for seg in unique_segmentations_up_to[from_this_char]:
                    unique_segmentations_up_to[to_this_char].append(seg + [from_this_char])

    # Part 2: Exponential  TODO: Seems to produce more segmentations than there actually are... Probably the inner two loops are fucked.
    if remaining == 0:
        yield from unique_segmentations_up_to[-1]
    else:
        text = original_text
        for prefix_length, segs_of_this_length in tqdm(enumerate(unique_segmentations_up_to), desc="Prefixes", total=len(unique_segmentations_up_to)):
            for segmentation in tqdm(segs_of_this_length, desc=f"Segmentations for prefix {prefix_length+1}"):
                segmentation = segmentation + [prefix_length]  # The whole point of segmentations_up_to is that the boundary at a given index is valid, but isn't included in the segmentation up to that index.
                for n_splits in range(remaining):
                    for c in itertools.combinations(range(max_prefix_length+1, len(text)), r=n_splits):  # A step to be in front of the character at max_prefix_length has already been attempted by the trellis, so we don't have to check for it anymore.
                        indices = segmentation + list(c)
                        tokens = segmentUsingIndices(text, indices)
                        if all(t in vocab for t in tokens):  # Some double work happens here (all the partial segmentations are already correct)
                            yield indices
