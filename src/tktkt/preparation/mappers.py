from abc import ABC, abstractmethod
import tokenizers.normalizers as tn
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class TextMapper(ABC):
    @abstractmethod
    def convert(self, text: str) -> str:
        pass


class InvertibleTextMapper(TextMapper):
    @abstractmethod
    def invert(self, text: str) -> str:
        pass


class Normaliser(TextMapper):

    def __init__(self, core: tn.Normalizer):
        self.hf = core

    def convert(self, text: str) -> str:
        return self.hf.normalize_str(text)


HUGGINGFACE_CHARMAP = bytes_to_unicode()
HUGGINGFACE_MAPCHAR = {v: k for k, v in HUGGINGFACE_CHARMAP.items()}

class ByteMapping(InvertibleTextMapper):

    def convert(self, text: str) -> str:
        return "".join(map(HUGGINGFACE_CHARMAP.get, text.encode("utf-8")))

    def invert(self, text: str) -> str:
        return bytes(map(HUGGINGFACE_MAPCHAR.get, text)).decode("utf-8")
