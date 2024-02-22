from typing import List, Callable
from dataclasses import dataclass

from ...interfaces.tokeniser import TokeniserWithVocab, Vocab
from ...preparation.splitters import WhitespaceAndMarkerPretokeniser, SpaceMarker


class ViterbiNode:
    def __init__(self, id=None):
        self.id = id
        self.current_loss = float("inf")
        self.backpointer  = None

    def __repr__(self):
        return f"N({self.id} -> {self.backpointer.id if self.backpointer else None})"


@dataclass
class ViterbiLoss:
    loss: float
    tiebreaker: float

    def __lt__(self, other):
        return (self.loss, self.tiebreaker) < (other.loss, other.tiebreaker)


@dataclass
class ViterbiLossUpdater:  # Doesn't use abstract methods because the implementation of the two methods is independent.
    loss_update: Callable[[ViterbiLoss, ...], float]
    tiebreaker_update: Callable[[ViterbiLoss, ...], float]

    def update(self, previous_loss: ViterbiLoss, *args, **kwargs) -> ViterbiLoss:
        return ViterbiLoss(self.loss_update(previous_loss, *args, **kwargs),
                           self.tiebreaker_update(previous_loss, *args, **kwargs))


class UnguidedViterbi(TokeniserWithVocab):
    """
    Subword tokeniser that uses a Viterbi algorithm to minimise the amount of tokens used
    for segmenting a word given a subword vocabulary.

    The subwords have no notion of probability, and we don't have sentence-level context.
    """

    def __init__(self, pretrained_vocabulary: Vocab, space_marker: SpaceMarker):
        super().__init__(preprocessor=WhitespaceAndMarkerPretokeniser(space_marker))
        self.vocab = pretrained_vocabulary

        self.updater = ViterbiLossUpdater(
            loss_update=lambda prev_loss, step_string: prev_loss.loss + 1,
            tiebreaker_update=lambda prev_loss, step_string: min(prev_loss.tiebreaker, -len(step_string))
        )
        # self.updater = ViterbiLossUpdater(
        #     loss_update=lambda prev_loss, step_string: -max(-prev_loss.loss, len(step_string)),
        #     tiebreaker_update=lambda prev_loss, _: 0
        # )

    # I just realised that instead of asking the vocab for a list of allowed substrings of a given string, I can also
    # just do an O(1) check whether each substring is allowed. Oops.
    #     self.trie = TrieNode()
    #     self.initialiseTrie()
    #
    # def initialiseTrie(self):
    #     for typ in self.vocab.keys():
    #         self.trie.add(typ)
    #     self.trie.compile()
    #     self.trie.compileRoots()

    def getVocab(self) -> Vocab:
        return self.vocab

    def tokenise(self, pretoken: str) -> List[str]:
        lattice = [ViterbiNode(i) for i in range(len(pretoken)+1)]  # Given a "string", this creates a Viterbi node at |s|t|r|i|n|g|.
        for node in lattice:
            node.current_loss = ViterbiLoss(float("inf"),0)
        lattice[0].current_loss = ViterbiLoss(0,0)

        # Forward pass
        for start_index, finished_node in enumerate(lattice):  # Technically you don't need the last iteration, but the inner loop is empty for it anyway.
            for stop_index in range(start_index+1, len(pretoken)+1):
                step = pretoken[start_index:stop_index]
                if step not in self.vocab:
                    continue
                proposed_loss = self.updater.update(finished_node.current_loss, step)

                destination = lattice[stop_index]
                if proposed_loss < destination.current_loss:  # First checks if loss is lower. If it's EQUAL, compares if tie-breaker is lower.
                    destination.backpointer = finished_node
                    destination.current_loss = proposed_loss

                    ### All paths that make it to the last node are printed
                    if stop_index == len(pretoken):
                        print(self.decodeNode(pretoken, lattice[-1]))
                    ###

        # Backward pass
        return self.decodeNode(pretoken, lattice[-1])

    def decodeNode(self, text: str, node: ViterbiNode):
        tokens = []
        latest_id    = node.id
        current_node = node.backpointer
        while current_node is not None:
            tokens.insert(0, text[current_node.id:latest_id])

            latest_id    = current_node.id
            current_node = current_node.backpointer
        return tokens


class RA_Product(TokeniserWithVocab):
    """
    The idea: we want to nicely distribute subwords over the string. One property of multiplication with a fixed sum is
    that it is highest when all numbers are as close as possible.
    Hence, this finds the segmentation with maximal product of subword lengths.

    Okay, in hindsight, it works, but it's too balanced. You might want to incorporate something into the
        score that nudges it towards fewer substrings.
        For a string of 6, it is true that
            1* 5*1 < 1* 6 = 1* 3*2*1 < 1* 4*2 = 1* 2*2*2 < 1* 3*3
        So the ordering of a product is... weird.
    """

    def tokenise(self, word: str) -> List[str]:
        scores = [0 for _ in range(len(word) + 1)]
        scores[0] = 1
        backpointer = [None for _ in range(len(word) + 1)]

        # Forward pass
        for start in range(len(word)):
            for end in range(start + 1, len(word) + 1):
                step = word[start:end]
                new_score = scores[start] * len(step)
                if step in self.vocab and new_score > scores[end]:
                    scores[end] = scores[start] * len(step)
                    backpointer[end] = start

        # Backward pass
        tokens = []
        current_idx = len(word) + 1
        next_idx = backpointer[-1]
        while next_idx is not None:
            tokens.append(word[next_idx:current_idx])
            current_idx = next_idx
            next_idx = backpointer[next_idx]
        tokens.reverse()
        return tokens
