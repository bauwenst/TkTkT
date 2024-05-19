from typing import List

import numpy.random as npr

from ...interfaces.tokeniser import TokeniserWithVocabDict, Preprocessor, Vocab


class RandomSegmentationFromVocab(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None):
        super().__init__(preprocessor, vocab, unk_type)
        self.rng = npr.default_rng(0)

    def generateSegmentations(self, text: str) -> List[List[int]]:
        """
        We have a function possibleSegmentations() somewhere that gives the AMOUNT of possible segmentations, but not
        what they are. A very similar Viterbi algorithm works here, except the history you store is not an integer but
        a collection of lists (where the amount computed at each step of the above algorithm is the size of that collection).

        A segmentation is represented by the list of start indices of its tokens.
        """
        unique_segmentations_up_to: List[List[List[int]]] = [[] for _ in range(len(text)+1)]  # Each element is a list of paths to get from "" to text[:index] (exclusive indexing).
        unique_segmentations_up_to[0].append([])
        for from_this_char in range(len(text)):  # The last index you can start from is the last character, len(pretoken)-1.
            for to_this_char in range(from_this_char+1, len(text)+1):  # The last index is len(pretoken), the exclusive upper bound for the entire string.
                if text[from_this_char:to_this_char] in self.vocab:
                    # For each segmentation that exists to the from_char, a new segmentation exists to the to_char now.
                    for seg in unique_segmentations_up_to[from_this_char]:
                        unique_segmentations_up_to[to_this_char].append(seg + [from_this_char])

        return [seg for seg in unique_segmentations_up_to[-1]]

    def _segmentUsingIndices(self, text: str, starts_of_tokens: List[int]) -> List[str]:
        return [text[start_idx:end_idx] for start_idx, end_idx in zip(starts_of_tokens, starts_of_tokens[1:] + [len(text)])]

    def tokenise(self, pretoken: str) -> List[str]:
        """
        Get a random segmentation that is possible using the tokeniser's vocabulary.
        """
        segmentations = self.generateSegmentations(pretoken)
        return self._segmentUsingIndices(pretoken, segmentations[self.rng.choice(len(segmentations))])
