from typing import List
from math import prod
import numpy.random as npr

from .generationbased import TokeniserWithVocabDict, Vocab, Preprocessor
from ...util.strings import segmentUsingBitmap, segmentUsingIndices


class RandomVocabSegmentation_RejectionSampling_BiasedBernoulli(TokeniserWithVocabDict):
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


from .graph import GraphTokeniser, ForwardGraphSampler, SegmentationGraph
class RandomVocabSegmentation_RejectionSampling_UniformGraph(GraphTokeniser):
    """
    Samples all segmentations with equal probability (i.e. uniformly) from the graph of all valid segmentations. Hence,
    the "needle-in-a-haystack" phenomenon does not happen here.

    First, a segmentation is sampled by sampling arcs from left to right in the graph, with each arc being equally likely.
    Then, depending on how likely that sample was, it is more or less likely to be rejected, in which case we retry. This
    way, the probability of emitting each segmentation is equal, even though the graph itself prefers some over others.

    This approach was described by Cognetta e.a. (2024) in https://aclanthology.org/2024.emnlp-main.600/.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab):
        super().__init__(preprocessor, vocab=vocab, sampler=ForwardGraphSampler())
        self.rng = npr.default_rng(0)

    def generateGraph(self, pretoken: str) -> SegmentationGraph:
        forepointers = [[] for _ in range(len(pretoken)+1)]
        for i in range(len(pretoken), 0, -1):  # Starts at index n, ends at index 1.
            for j in range(i-1, -1, -1):  # Last index is 0.
                token = pretoken[j:i]
                if self.hasType(token):  # To get from BEFORE character i to BEFORE character j, you do include j and don't include i.
                    forepointers[j].append(i)

        out_degree = [len(b) for b in forepointers]
        probabilities = [ [1/out_degree[node]]*out_degree[node] if out_degree[node] else []  # 1/d_o(n) for each backpointer.
                         for node in range(len(pretoken)) ]
        return SegmentationGraph(pointers=forepointers, probabilities=probabilities)

    def tokenise(self, pretoken: str) -> List[str]:
        # Generate the graph first.
        graph = self.generateGraph(pretoken)

        # Rejection sampling:
        #   - Determine epsilon, i.e. the probability of the possibly non-existent path that does all possible samplings (it visits all nodes where probability could be added to the final product).
        eps = 1/prod(len(ps) or 1   for ps in graph.pointers)

        #   - Generate valid paths and retry based on eps and its probability.
        while True:
            indices, p = self.sampler.samplePathAndProb(graph)
            if self.rng.random() < eps/p:  # Probability of being emitted is P(generate)*P(accept) = p*eps/p = eps. To verify the direction of "<", note that when P(accept) = eps/p = 1, you must always accept the path, and indeed, rand() is always lower than 1.
                print("emitted")
                return segmentUsingIndices(pretoken, starts_of_tokens=indices)
