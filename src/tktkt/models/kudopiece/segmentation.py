from pathlib import Path
from typing import List, Optional

from sentencepiece import SentencePieceProcessor

from ...interfaces.tokeniser import TokeniserWithVocabDict, Preprocessor, Vocab
from .vocabularisation import KudoPieceTrainer


class KudoPieceTokeniser(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, model_file: Path, kbest: int=1, smoothing_power: float=1,
                 vocab: Optional[Vocab]=None, special_tokens: Optional[List[str]]=None):
        """
        :param kbest: KudoPiece finds the k segmentations s with the highest joint token probabilities P(s)...
        :param smoothing_power: ...and samples across those k proportionally to P(s)^smoothing_power.
        :param vocab: Predefined vocabulary you want to use. If not given, it will be loaded from the same folder as the model file.
        :param special_tokens: If a vocab is not given, the special tokens to add when we load the vocab automatically.
        """
        self._k = kbest
        self._alpha = smoothing_power

        self.core = SentencePieceProcessor()
        self.core.Init(model_file.as_posix())
        if vocab is None:
            vocab = KudoPieceTrainer.load(model_file.with_suffix(".vocab"), existing_types=special_tokens)
        super().__init__(preprocessor, vocab=vocab)

    def tokenise(self, pretoken: str) -> List[str]:
        tokens = self.core.EncodeAsPieces(pretoken, enable_sampling=True, nbest_size=self._k, alpha=self._alpha)
        return tokens
