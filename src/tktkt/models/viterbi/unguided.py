from typing import List, Callable
from dataclasses import dataclass

from ...interfaces.general import TokeniserWithVocab, Vocab
from ...preparation.splitters import WordSplitter, SpaceMarker
from .lattice import *
from .trie import TrieNode


class UnguidedViterbi(TokeniserWithVocab):
    """
    Subword tokeniser that uses a Viterbi algorithm to minimise the amount of tokens used
    for segmenting a word given a subword vocabulary.

    The subwords have no notion of probability, and we don't have sentence-level context.
    """

    def __init__(self, pretrained_vocabulary: Vocab, space_marker: SpaceMarker):
        super().__init__(pretokeniser=WordSplitter(space_marker))
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
