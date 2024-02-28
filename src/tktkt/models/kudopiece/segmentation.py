from pathlib import Path
from typing import List

from sentencepiece import SentencePieceProcessor
from bpe_knockout.datahandlers.wordfiles import iterateWordsFile

from ...interfaces.tokeniser import TokeniserWithVocab, Preprocessor, Vocab


class KudoPieceTokeniser(TokeniserWithVocab):

    def __init__(self, preprocessor: Preprocessor, model_file: Path):
        self.core = SentencePieceProcessor()
        self.core.Init(model_file.as_posix())
        super().__init__(preprocessor, vocab=KudoPieceTokeniser.vocabFromFile(model_file.with_suffix(".vocab")))

    def tokenise(self, pretoken: str) -> List[str]:
        tokens = self.core.EncodeAsPieces(pretoken, enable_sampling=False)
        return tokens

    @staticmethod
    def vocabFromFile(vocab_file: Path) -> Vocab:
        with open(vocab_file, "r", encoding="utf-8") as handle:
            return {t: i for i,t in enumerate(typ for typ,_ in iterateWordsFile(handle, sep="\t"))}