from typing import List

from bpe_knockout.knockout.core import BTE, BteInitConfig

from ...interfaces.general import TokeniserWithVocab, Vocab
from ...preparation.splitters import WordSplitter


class BpeTokeniser(TokeniserWithVocab):

    def __init__(self, pretokeniser: WordSplitter, vocab: Vocab, merges: List[str]):
        super().__init__(pretokeniser, vocab)
        self.core = BTE(BteInitConfig(), starting_vocab=vocab, starting_mergelist=merges)

    def tokenise(self, pretoken: str) -> List[str]:
        assert isinstance(self.pretokeniser, WordSplitter)

        # I'm going to absolutely abuse the interface of my own library to pass an iterable to the segmentation
        # algorithm instead of a string, allowing any pretokeniser to be used. Based.
        return self.core.segment_as_is(self.pretokeniser.splitWord(pretoken))
