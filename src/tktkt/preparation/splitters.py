"""
Pretokenisation, i.e. splitting text into the units that will be tokenised separately.
"""
from typing import List, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from tokenizers import Regex
from tokenizers import pre_tokenizers as tp
from tokenizers import decoders as td
from transformers import PreTrainedTokenizerFast
from string import punctuation as BASIC_PUNCTUATION
import re
import regex  # Has \p{} classes

from .mappers import InvertibleTextMapper
from ..preparation.spacemarking import SpaceMarker, SpaceMarkerLocation


class Pretokeniser(ABC):
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """
        Split an entire text (e.g. a sentence) into smaller pre-token strings.
        It is these pre-token strings that will SEPARATELY be passed to the tokeniser.
        """
        pass

    @abstractmethod
    def invertToken(self, token: str) -> str:
        """
        Takes (part of) a pretoken and undoes any character transformations that were applied during pretokenisation.
        Tricky, because this isn't necessarily possible: for example, if you mapped from 1 Unicode charactere to >1 bytes,
        and the tokeniser separated those bytes into separate tokens, converting one token at a time will not work.

        TODO: You should probably have a flag that decides whether you should join inverted tokens or invert joined tokens.
        """
        pass

    def unsplit(self, tokens: List[str]) -> str:
        """
        Inverts the splitting operation.
        """
        return "".join(map(self.invertToken, tokens))


class PretokeniserSequence(Pretokeniser):
    def __init__(self, sub_pretokenisers: List[Pretokeniser]):
        self.sequence = sub_pretokenisers

    def split(self, text: str) -> List[str]:
        current_pretokens = [text]
        for pretokeniser in self.sequence:
            generated_pretokens = []
            for pretoken in current_pretokens:
                generated_pretokens.extend(pretokeniser.split(pretoken))

            current_pretokens = generated_pretokens

        return current_pretokens

    def invertToken(self, token: str) -> str:
        """
        TODO: Clearly this isn't how you should invert a sequence of split-then-map transformations.
        """
        for pretokeniser in reversed(self.sequence):
            token = pretokeniser.invertToken(token)
        return token


class IdentityPretokeniser(Pretokeniser):

    def split(self, text: str) -> List[str]:
        return [text]

    def invertToken(self, token: str) -> str:
        return token


class PunctuationPretokeniser(Pretokeniser):
    """
    Split on punctuation and conserve it.

    You should run this BEFORE splitting on spaces. There is an argument to be made to run pretokenisers in the
    sequence "split on spaces -> add SoW/EoW -> split on punctuation". The big issue with this is that it confuses
    whitespace with semantics: SoW/EoW is NOT a space marking, but a marking to signal that a new word has begun. If
    you are using a SoW, then in the text

        "Hello world (world being our planet)!"

    the second "world" should STILL receive a SoW because it does contain the start of a new word even if it is not
    preceded by a space. What matters to any downstream model is the former, not the latter.
    """

    class HyphenMode(Enum):
        ONLY     = 1
        EXCLUDED = 2
        INCLUDED = 3

    def __init__(self, hyphen_mode: "PunctuationPretokeniser.HyphenMode", grab_preceding_space: bool=False):
        punctuation_hyphens_only = "-–_"
        if hyphen_mode == PunctuationPretokeniser.HyphenMode.ONLY:
            punctuation = punctuation_hyphens_only
        else:
            punctuation = BASIC_PUNCTUATION + "€£…‘’“”„«»"      # Add some European punctuations.
            punctuation = punctuation.replace("\\", "") + "\\"  # Put backslash in the back. Makes the pattern clearer.
            for hyphen in punctuation_hyphens_only:
                punctuation = punctuation.replace(hyphen, "")  # Note: Not all of these will have effect, but some will.

            if hyphen_mode == PunctuationPretokeniser.HyphenMode.INCLUDED:
                for hyphen in punctuation_hyphens_only:
                    punctuation = hyphen + punctuation  # Note: All of these will have effect.

        punctuation_escaped = punctuation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("-", "\\-")
        self.core = tp.Split(pattern=Regex(" ?"*grab_preceding_space + "[" + punctuation_escaped + "]+"), behavior="isolated")

    def split(self, text: str) -> List[str]:
        return [w for w, _ in self.core.pre_tokenize_str(text)]

    def invertToken(self, token: str) -> str:
        return token


class WhitespaceAndMarkerPretokeniser(Pretokeniser):
    """
    Splits on (and DESTROYS) consecutive whitespace, replacing it by a marker.

    A general principle I adhere to is that the addition of space markings should NOT be signalled by the user text.
    For example, the user should not have to put a space in front of his input to make the tokeniser put a marker
    in front of the input.
    If the user wants to control, for the same tokeniser, if a marker is added, it should be an argument to
    the .prepareAndTokenise() method.
    """

    def __init__(self, replacement: SpaceMarker):
        self.marker = replacement

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []

        pretokens = [text]
        if self.marker.location == SpaceMarkerLocation.TOKEN:
            pretokens = text.split()  # Will strip all whitespace from both sides, then split on any span of whitespace.
            pretokens = WhitespaceAndMarkerPretokeniser.intercalate(pretokens, self.marker.substitute)  # Will have length 2n-1.
            if text[0].isspace():
                pretokens.insert(0, self.marker.substitute)
            if text[-1].isspace():  # Due to the sanity check above, we know that this is not the same whitespace!
                pretokens.append(self.marker.substitute)

        elif self.marker.location == SpaceMarkerLocation.START:
            pretokens = text.split()
            for i in range(len(pretokens)):
                if i != 0 or text[0].isspace():
                    pretokens[i] = self.marker.substitute + pretokens[i]

        elif self.marker.location == SpaceMarkerLocation.END:
            pretokens = text.split()
            for i in range(len(pretokens)):
                if i != len(pretokens)-1 or text[-1].isspace():
                    pretokens[i] = pretokens[i] + self.marker.substitute

        return pretokens

    def splitWord(self, pretoken: str) -> List[str]:
        """
        Extra method for algorithms like BPE that require a word to start out as
        being split into characters.

        Does NOT add SoW/EoW because this is already done when you split the sentence into words.
        What it might do, however, is attach the SoW/EoW to the adjacent character.
        """
        if self.marker.location == SpaceMarkerLocation.START:
            chars, sow = self.stripMarker(pretoken)
            if self.marker.detached:
                return [sow] + list(chars)
            else:
                return [sow + chars[0]] + list(chars[1:])
        elif self.marker.location == SpaceMarkerLocation.END:
            chars, eow = self.stripMarker(pretoken)
            if self.marker.detached:
                return list(chars) + [eow]
            else:
                return list(chars[:-1]) + [chars[-1] + eow]
        elif self.marker.location == SpaceMarkerLocation.TOKEN:
            return list(pretoken)
        else:
            return [pretoken]

    def stripMarker(self, pretoken: str) -> Tuple[str,str]:
        """
        Retrieve the part of a pretoken that isn't a space marker.
        """
        L = len(self.marker.substitute)
        if self.marker.location == SpaceMarkerLocation.START:
            root, marker = pretoken[L:], pretoken[:L]
        elif self.marker.location == SpaceMarkerLocation.END:
            root, marker = pretoken[:len(pretoken)-L], pretoken[len(pretoken)-L:]
        elif self.marker.location == SpaceMarkerLocation.TOKEN:
            root, marker = pretoken, ""
        else:
            root, marker = pretoken, ""

        return root, marker

    def invertToken(self, token: str) -> str:
        return token.replace(self.marker.substitute, " ")  # TODO: Technically should not do replacements in the middle.

    @staticmethod
    def intercalate(lst: list, new_element):
        new_list = []
        for old_element in lst:
            new_list.append(old_element)
            new_list.append(new_element)
        return new_list[:-1]


class WhitespacePretokeniser(Pretokeniser):
    """
    Slightly broader splitter than Python's native .split() since this also reacts to word separators that aren't
    visible to the naked eye, e.g. ZWSP.
    """

    def __init__(self, destructive: bool=True):
        if destructive:
            self.pattern = re.compile(r"[\s​]+")
        else:
            self.pattern = re.compile(r"([\s​]+)")

    def split(self, text: str) -> List[str]:
        pretokens = self.pattern.split(text)
        return [t for t in pretokens if t]

    def invertToken(self, token: str) -> str:
        return token


class IsolateDigits(Pretokeniser):
    """
    Isolate all digits into single-character tokens. If you don't know why we need this, read
    https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/

    Works for any language and any numerical character. Try e.g. ૪ (4 in Gujarati) or ௲ (1000 in Tamil) or ½.
    Reference: https://character.construction/numbers
    """

    def __init__(self):
        self.pattern = regex.compile(r"(\p{N})")

    def split(self, text: str) -> List[str]:
        pretokens = self.pattern.split(text)
        return [t for t in pretokens if t]


class AddWordBoundary(Pretokeniser):
    """
    Does not split text, but only adds a space marker, assuming the text in its entirety needs only one.

    Is entirely insensitive to the presence of spaces. Using spaces as boundary markers is a bad idea for multiple
    reasons, among which the fact that a text doesn't always start (for SoW) or end (for EoW) with a space, and that
    punctuation sometimes removes preceding or succeeding space. A boundary is a boundary.
    """

    def __init__(self, marker: SpaceMarker):
        self.marker = marker

    def split(self, text: str) -> List[str]:
        if self.marker.location == SpaceMarkerLocation.TOKEN:
            return [self.marker.substitute, text]
        elif self.marker.location == SpaceMarkerLocation.START:
            return [self.marker.substitute + text]
        elif self.marker.location == SpaceMarkerLocation.END:
            return [text + self.marker.substitute]
        else:
            return [text]

    def invertToken(self, token: str) -> str:
        return token.replace(self.marker.substitute, " ")  # TODO: Technically should not do replacements in the middle.


class MapperAsPretokeniser(Pretokeniser):
    """
    When used in a sequence of pretokenisers, this allows you to apply a text->text mapping on individual pretokens
    produced by another pretokeniser. In HuggingFace, for example, you can't do this.
    """

    def __init__(self, mapper: InvertibleTextMapper):
        self.core = mapper

    def split(self, text: str) -> List[str]:
        return [self.core.convert(text)]

    def invertToken(self, token: str) -> str:
        return self.core.invert(token)


class HuggingFacePretokeniser(Pretokeniser):

    def __init__(self, hf_model: PreTrainedTokenizerFast):
        """
        Steals the pretokeniser from a HuggingFace tokeniser.
        Only possible for the "Fast" variants because some people don't know how to design a software system.
        https://github.com/huggingface/transformers/issues/26254
        """
        self.encode: tp.PreTokenizer = hf_model.backend_tokenizer.pre_tokenizer
        self.decode: td.Decoder      = hf_model.backend_tokenizer.decoder

    def split(self, text: str) -> List[str]:
        return [w for w, _ in self.encode.pre_tokenize_str(text)]

    def invertToken(self, token: str) -> str:
        return self.decode.decode([token])
