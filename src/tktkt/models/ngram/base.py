"""
HuggingFace-compliant character-N-gram and byte-N-gram tokenisers.

TODO:
      - You need to add [UNK] in character-based mode when something outside the vocabulary appears.
        The HuggingFace wrapper for the BTE tokeniser does this based on whether a given token string is in the vocab,
        but we have no vocab for N-gram tokenisers.
      - It's possible that a true N-gram tokeniser needs padding characters so that all tokens are exactly N long rather
        than allowing the tail to be 1...N characters.
"""
from enum import Enum

from copy import copy

from bpe_knockout.util.datahandlers.hf_corpora import punctuation
from bpe_knockout.util.bytemapping import BYTE_ALPHABET

from ...interfaces.tokenisers import *
from ...preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from ...factories.preprocessors import Preprocessor, IdentityMapper, PseudoByteMapping, IsolatePunctuation, OnWhitespace, IsolatedSpaceMarker, PretokeniserSequence, AddWordBoundary, MapperAsPretokeniser, HyphenMode


LETTERS = {chr(i) for i in range(97,123)} | {chr(i) for i in range(65,91)} \
        | {chr(i) for i in range(224,229)} | {chr(i) for i in range(232,240)} | {chr(i) for i in range(249,253)} | {chr(i) for i in range(242,247)} \
        | {"ñ", "œ", "ç", "ẞ", "å", "ø" }
ASCII_PUNCTUATION = {char for char in punctuation if ord(char) < 256}


class NgramByteBasedMode(Enum):
    CHAR_NGRAMS = 0
    BYTE_NGRAMS = 1  # Sometimes these tokens cannot all be decoded into characters individually, unlike the other two modes.
    CHAR_NGRAMS_AS_BYTES = 2


class NgramTokeniser(Tokeniser):
    """
    Byte/character N-gram tokeniser.

    NOTE: If you ever want to use this in a model for extrinsic evaluation, beware that this tokeniser uses
          Huck-like space marking, i.e. a separate token indicates that there is a space. It is never part of
          another token. This is important to know if you want a fair comparison between tokenisers.
    """

    def __init__(self, N: int, mode: NgramByteBasedMode=NgramByteBasedMode.CHAR_NGRAMS, word_boundary: BoundaryMarker=IsolatedSpaceMarker):
        if N < 1 or not isinstance(N, int):
            raise ValueError("N-gram tokenisers only exist for N = 1, 2, 3, ...")
        self.N = N
        self.mode = mode

        self.marker  = copy(word_boundary)
        self.marker.detached = False
        self.bytemap = PseudoByteMapping()
        preprocessor = Preprocessor(
            IdentityMapper(),
            IdentityMapper(),
            PretokeniserSequence([
                OnWhitespace(destructive=True),
                IsolatePunctuation(HyphenMode.EXCLUDED),
                MapperAsPretokeniser((self.bytemap if NgramByteBasedMode.BYTE_NGRAMS else IdentityMapper())),
                AddWordBoundary(self.marker),
                IsolatePunctuation(HyphenMode.ONLY)
            ])
        )

        if self.mode != NgramByteBasedMode.CHAR_NGRAMS:
            self.alphabet = (set(BYTE_ALPHABET) - ASCII_PUNCTUATION, ASCII_PUNCTUATION)  # These two sets can never appear in the same token.
        else:
            self.alphabet = (LETTERS, set(punctuation))

        super().__init__(preprocessor=preprocessor)

    def tokenise(self, pretoken: str) -> Tokens:
        pretoken, marker = self.marker.isolate(pretoken)
        if not pretoken:
            return [marker]

        tokens = [pretoken[i*self.N:(i+1)*self.N] for i in range((len(pretoken)-1)//self.N + 1)]

        if self.mode == NgramByteBasedMode.CHAR_NGRAMS_AS_BYTES:
            tokens_as_bytes = []
            for t in tokens:
                tokens_as_bytes.append(self.bytemap.convert(t))
            tokens = tokens_as_bytes

        if self.marker.location == BoundaryMarkerLocation.START:
            tokens[0] = marker + tokens[0]
        elif self.marker.location == BoundaryMarkerLocation.END:
            tokens[-1] = tokens[-1] + marker

        return tokens

    def getName(self):
        return f"{self.N}-gram" if self.N != 1 else "Char"
