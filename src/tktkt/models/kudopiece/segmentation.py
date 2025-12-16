from pathlib import Path
from typing import Optional

from sentencepiece import SentencePieceProcessor

from ...interfaces.tokenisers import *
from .vocabularisation import KudoPieceVocabulariser


class KudoPieceTokeniser(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, model_file: Path, kbest: int=1, smoothing_power: float=1,
                 vocab: Optional[Vocab[WithSpecials]]=None, special_tokens: Optional[WithSpecials]=None):
        """
        :param kbest: KudoPiece finds the k segmentations s with the highest joint token probabilities P(s)...
        :param smoothing_power: ...and samples across those k proportionally to P(s)^smoothing_power.
        :param vocab: Predefined vocabulary you want to use. If not given, it will be loaded from the same folder as the model file.
        :param special_tokens: If a vocab is not given, the special tokens to add when we load the vocab automatically.
        """
        self._k = kbest
        self._alpha = smoothing_power
        self._is_stochastic = kbest > 1 or kbest == -1

        self.core = SentencePieceProcessor()
        self.core.Init(model_file.as_posix())
        if vocab is None:
            vocab = KudoPieceVocabulariser.load(model_file.with_suffix(".vocab"), specials=special_tokens)
        super().__init__(preprocessor=preprocessor, vocab=vocab)

    def tokenise(self, pretoken: str) -> Tokens:
        tokens = self.core.EncodeAsPieces(pretoken, enable_sampling=self._is_stochastic, nbest_size=self._k, alpha=self._alpha)
        return tokens

    def getName(self) -> str:
        return f"KudoPiece(ℓ={self._k},α={self._alpha})"
