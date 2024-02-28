"""
Interfaces to make tokenisers that are compatible with the HuggingFace family.

Requires you to implement all missing methods of transformers.PreTrainedTokenizer.
"""
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from transformers import PreTrainedTokenizer


class HuggingFaceTokeniserProtocol(ABC, PreTrainedTokenizer):

    @abstractmethod
    def _tokenize(self, text, **kwargs):
        pass

    # The following three methods are for interfacing with the vocabulary.

    @property  # Property because that's how HuggingFace does it. Makes no sense to have getter/setter for this, but ok.
    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def _convert_token_to_id(self, token: str):
        pass

    @abstractmethod
    def _convert_id_to_token(self, index: int):
        pass

    # The following two methods are for storage and come from the parent class of PreTrainedTokenizer.

    @abstractmethod
    def get_vocab(self):
        pass

    @abstractmethod
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        pass

    # The following two methods are technically already implemented in HF, but it's important to define them explicitly.

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return super().tokenize(text, **kwargs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return super().convert_tokens_to_string(tokens)
