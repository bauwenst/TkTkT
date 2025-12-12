"""
HuggingFace-compliant character-N-gram and byte-N-gram tokenisers.
"""
from typing import List
import re

from .base import NgramTokeniser, NgramByteBasedMode
from ...interfaces.huggingface import HuggingFaceTokeniserInterface


from bpe_knockout.util.datahandlers.hf_corpora import punctuation

LETTERS = {chr(i) for i in range(97,123)} | {chr(i) for i in range(65,91)} \
        | {chr(i) for i in range(224,229)} | {chr(i) for i in range(232,240)} | {chr(i) for i in range(249,253)} | {chr(i) for i in range(242,247)} \
        | {"ñ", "œ", "ç", "ẞ", "å", "ø" }
ASCII_PUNCTUATION = {char for char in punctuation if ord(char) < 256}


class NgramTokeniser_Hf(HuggingFaceTokeniserInterface):
    """
    NOTE: If you ever want to use this in a model for extrinsic evaluation, beware that this tokeniser uses
      Huck-like space marking, i.e. a separate token indicates that there is a space. It is never part of
      another token. This is important to know if you want a fair comparison between tokenisers.
    """

    def __init__(self, backend: NgramTokeniser):
        super().__init__()
        self.backend = backend

    @property
    def vocab_size(self):
        """
        Theoretically, assuming no pretokenisation:
            - If it's byte-based, there are 256^N of the biggest possible tokens.
            - If you support all Unicode, it's (149 878)^N.
            - If you want European languages, it's (26 + ...)^N, where the ... depends on
                - whether you include uppercase letters
                - whether you including accents (é, è, ë, ê, ...)
                - whether you include special characters like œ, ç, ẞ, å, ø, ñ, ...

        Practically, because punctuation and letters are N-grammed separately, you get a smaller vocabulary.
        However, because word lengths aren't perfect multiples of N, smaller subwords are also used, increasing the vocabulary.

        There's also 1 space token.

        If you do N-char splits (instead of N-byte) and convert to UTF-8 bytes afterwards, you can have up to 4N bytes
        in the biggest subwords, making the total vocabulary something like sum_{i=1}^{4N} 256^{i}.
        """
        if self.backend.mode != NgramByteBasedMode.CHAR_NGRAMS_AS_BYTES:
            return 1 + sum(len(self.alphabet[0]) ** i for i in range(1, self.N+1)) \
                     + sum(len(self.alphabet[1]) ** i for i in range(1, self.N+1))
        else:
            return 1 + sum(len(self.alphabet[0]) ** (4*i) for i in range(1, self.N+1)) \
                     + sum(len(self.alphabet[1]) ** (4*i) for i in range(1, self.N+1))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return re.sub(r"(?<! ) ", "", self.backend.preprocessor.undo(tokens))


# def list_split(lst: list, sep):
#     sublists = []
#     current_sublist = []
#     for item in lst:
#         if item == sep:
#             sublists.append(current_sublist)
#             current_sublist = []
#         else:
#             current_sublist.append(item)
#
#     sublists.append(current_sublist)
#     return sublists
