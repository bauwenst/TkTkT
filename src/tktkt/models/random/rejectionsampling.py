from typing import List
import numpy.random as npr

from .generationbased import TokeniserWithVocabDict, Vocab, Preprocessor
from ...util.strings import segmentUsingBitmap, segmentUsingIndices


class RandomVocabSegmentation_RejectionSampling(TokeniserWithVocabDict):
    """
    Splits the given string at random positions and checks if all resulting tokens are in the given vocab.
    Retries until the answer is yes.
    Biased towards not segmenting because otherwise you mostly get small-token segmentations.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, unk_type: str=None):
        super().__init__(preprocessor, vocab, unk_type)
        self.rng = npr.default_rng(0)

    def tokenise(self, pretoken: str) -> List[str]:
        return rejectionSamplingBiased(pretoken, self.vocab, self.rng)


def rejectionSampling(text: str, vocab: Vocab, rng: npr.Generator):
    """
    Sample random segmentations until one is valid.

    This may require needle-in-a-haystack luck, and is subject to adversarial attacks (e.g. words with typos have fewer
    valid segmentations). A fully randomised string will take impossibly long to segment.
    """
    n_positions = len(text)-1
    n_segmentations = 2**n_positions
    while True:
        segmentation_index = rng.integers(n_segmentations)
        bitmap = bin(segmentation_index)[2:].zfill(n_positions)
        segmentation = segmentUsingBitmap(text, bitmap)
        if all(token in vocab for token in segmentation):
            return segmentation


def rejectionSamplingBiased(text: str, vocab: Vocab, rng: npr.Generator,
                            max_tries: int=5_000) -> List[str]:
    """
    Sample random segmentations until one is valid, where single-character tokens are disincentivised by having split
    positions be less likely than non-split positions.

    This may require even more needle-in-a-haystack luck.
    """
    n_positions = len(text)-1

    backoff = [0.05, 0.1, 0.25]  # Expect p = 0.005 to cost about 50k tries at most, and p = 0.01 about 10k. Every loop of 1000 tries is about 2 ms. We want to keep tokenisers to at least 200 tkz/s, which is 0.005 s/tkz = 5 ms/tkz.
    tries = 0
    for p in backoff:
        tries += 1  # Technically adds too much, but you have to add 1 to get the while loop to be re-entered.
        while tries % max_tries != 0:
            indices = rng.binomial(n=1, p=p, size=n_positions).nonzero()[0].tolist()
            segmentation = segmentUsingIndices(text, [0] + indices)
            if all(token in vocab for token in segmentation):
                # print(tries)
                return segmentation
            tries += 1

    return list(text)
