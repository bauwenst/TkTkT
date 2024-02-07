"""
HuggingFace-compliant character-N-gram and byte-N-gram tokenisers.

TODO:
      - You need to add [UNK] in character-based mode when something outside the vocabulary appears.
        The HuggingFace wrapper for the BTE tokeniser does this based on whether a given token string is in the vocab,
        but we have no vocab for N-gram tokenisers.
      - It's possible that a true N-gram tokeniser needs padding characters so that all tokens are exactly N long rather
        than allowing the tail to be 1...N characters.
"""
from typing import List
from enum import Enum
import re

from bpe_knockout.auxiliary.tokenizer_interface import BasicStringTokeniser
from bpe_knockout.auxiliary.bytemapping import BYTE_ALPHABET
from bpe_knockout.datahandlers.hf_corpora import punctuation_regex_str, punctuation
from bpe_knockout.knockout.core import ByteBasedMode

SPACE_TYPE = "[SPACE]"
Punctuation = re.compile("(" + punctuation_regex_str + ")")

LETTERS = {chr(i) for i in range(97,123)} | {chr(i) for i in range(65,91)} \
        | {chr(i) for i in range(224,229)} | {chr(i) for i in range(232,240)} | {chr(i) for i in range(249,253)} | {chr(i) for i in range(242,247)} \
        | {"ñ", "œ", "ç", "ẞ", "å", "ø" }
ASCII_PUNCTUATION = {char for char in punctuation if ord(char) < 256}


class NgramByteBasedMode(Enum):
    CHAR_NGRAMS = 0
    BYTE_NGRAMS = 1  # Sometimes these tokens cannot all be decoded into characters individually, unlike the other two modes.
    CHAR_NGRAMS_AS_BYTES = 2


class NgramTokeniser(BasicStringTokeniser):
    """
    Byte/character N-gram tokeniser.

    NOTE: If you ever want to use this in a model for extrinsic evaluation, beware that this tokeniser uses
          Huck-like space marking, i.e. a separate token indicates that there is a space. It is never part of
          another token. This is important to know if you want a fair comparison between tokenisers.
    """

    def __init__(self, N: int, mode: NgramByteBasedMode=NgramByteBasedMode.CHAR_NGRAMS):
        if N < 1:
            raise ValueError("N-gram tokenisers only exist for N = 1, 2, 3, ...")
        self.N = N
        self.mode = mode

        do_bytes = mode != NgramByteBasedMode.CHAR_NGRAMS
        self._word_preprocessor, self._tokens_to_word = (ByteBasedMode.INPUT_TO_BYTES if do_bytes else ByteBasedMode.NONE).toInputProcessors()

        if do_bytes:
            self.alphabet = (set(BYTE_ALPHABET) - ASCII_PUNCTUATION, ASCII_PUNCTUATION)  # These two sets can never appear in the same token.
        else:
            self.alphabet = (LETTERS, set(punctuation))

    def getName(self):
        return f"{self.N}-gram" if self.N != 1 else "Char"

    @property
    def vocab_size(self):
        """
        Theoretically, assuming no pretokenisation:
            - If it's byte-based, there are 256^N of the biggest possible tokens.
            - If you support all Unicode, it's (149 186)^N.
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
        if self.mode != NgramByteBasedMode.CHAR_NGRAMS_AS_BYTES:
            return 1 + sum(len(self.alphabet[0]) ** i for i in range(1, self.N+1)) + sum(len(self.alphabet[1]) ** i for i in range(1, self.N+1))
        else:
            return 1 + sum(len(self.alphabet[0]) ** (4*i) for i in range(1, self.N+1)) + sum(len(self.alphabet[1]) ** (4*i) for i in range(1, self.N+1))

    def tokenize(self, text: str) -> List[str]:
        words = text.split()  # this is a certified Ada Wan moment
        tokens = []
        for spaceless_word in words:
            for word_or_punctuation in Punctuation.split(spaceless_word):  # The reason to split on punctuation FIRST, rather than converting to bytes, is to prevent HuggingFace bytes being interpreted as punctuation. It's slower, but it's correct.
                if self.mode == NgramByteBasedMode.BYTE_NGRAMS:
                    word_or_punctuation = self._word_preprocessor(word_or_punctuation)

                new_tokens = [word_or_punctuation[i*self.N:(i+1)*self.N] for i in range((len(word_or_punctuation)-1)//self.N + 1)]
                if self.mode == NgramByteBasedMode.CHAR_NGRAMS_AS_BYTES:
                    new_tokens = [self._word_preprocessor(token) for token in new_tokens]

                tokens.extend(new_tokens)
            tokens.append(SPACE_TYPE)

        return tokens[:-1]

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(self._tokens_to_word(word_tokens)
                        for word_tokens in list_split(tokens, SPACE_TYPE))


def list_split(lst: list, sep):
    sublists = []
    current_sublist = []
    for item in lst:
        if item == sep:
            sublists.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.append(item)

    sublists.append(current_sublist)
    return sublists


if __name__ == "__main__":
    from itertools import product
    from bpe_knockout.auxiliary.tokenizer_interface import tokenizeAsWord
    sentence = "Life could be a drëam!"
    word = "supercälifragïlisticëxpialidocious"

    Ns = [2,3,4,5,10]
    modes = [NgramByteBasedMode.CHAR_NGRAMS, NgramByteBasedMode.CHAR_NGRAMS_AS_BYTES, NgramByteBasedMode.BYTE_NGRAMS]
    for n,m in product(Ns, modes):
        print(n,m)
        tk = NgramTokeniser(n, m)
        print(tk.tokenize(sentence))
        print(tokenizeAsWord(sentence, tk))
        # print(tk.tokenize(word))
        # print(tokenizeAsWord(word, tk))
        print()