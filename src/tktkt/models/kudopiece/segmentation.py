from pathlib import Path
from typing import List

from sentencepiece import SentencePieceProcessor

from ...interfaces.tokeniser import TokeniserWithVocabDict, Preprocessor
from .vocabularisation import KudoPieceTrainer


class KudoPieceTokeniser(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, model_file: Path):
        self.core = SentencePieceProcessor()
        self.core.Init(model_file.as_posix())
        super().__init__(preprocessor, vocab=KudoPieceTrainer.load(model_file.with_suffix(".vocab"), sorting_key=None, existing_types=None))

    def tokenise(self, pretoken: str) -> List[str]:
        tokens = self.core.EncodeAsPieces(pretoken, enable_sampling=False)
        return tokens
