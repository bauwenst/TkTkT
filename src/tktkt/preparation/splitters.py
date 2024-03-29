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

from ..preparation.spacemarking import SpaceMarker, SpaceMarkerLocation


class Pretokeniser(ABC):
    """
    A note on the design of pretokenisers: you want pretokenisers to be invertible to get back to a single string, but
    especially with sequentially applied pretokenisers, this is ill-defined (because fundamentally, list.extend isn't
    invertible). There are three ways you can implement un-splitting:

            1. "".join(map(invertOne, tokens)) or
            2. "".join(invertMany(tokens)) or
            3. invertString("".join(tokens))

    This matters because for example, the ALBERT ULM decoder is something like "".join(tokens).replace("_", " ").strip(),
    which means that inverting single tokens can never produce spaces. Spaces are only protected when they appear
    in tokens that are surrounded by other tokens during decoding. This is an example of an invertMany, like all HuggingFace pretokenisers.

    1 and 3 are special cases of 2, where respectively
        invertMany(tokens) == map(invertOne, tokens)
        invertMany(tokens) == [invertString("".join(tokens))]
    A class may elect to use such implementations.
    """

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """
        Split an entire text (e.g. a sentence) into smaller pre-token strings.
        It is these pre-token strings that will SEPARATELY be passed to the tokeniser.
        """
        pass

    @abstractmethod
    def invertTokens(self, pretokens: List[str]) -> List[str]:
        """
        Invert any string transformations applied in the process of splitting a string into pretokens.
        May also apply a merging operation into a smaller list if that is appropriate.
        """
        pass  # For example, [invertToken(t) for t in tokens], but you should make this explicit because it's not necessarily the case.

    def invertToken(self, pretoken: str) -> str:
        """
        Takes (part of) a pretoken and undoes any character transformations that were applied during pretokenisation.
        Tricky, because this isn't necessarily possible: for example, if you mapped from 1 Unicode charactere to >1 bytes,
        and the tokeniser separated those bytes into separate tokens, converting one token at a time will not work.
        """
        return "".join(self.invertTokens([pretoken]))

    def unsplit(self, tokens: List[str]) -> str:
        """
        Inverts the splitting operation.
        """
        return "".join(self.invertTokens(tokens))

    def __call__(self):  # Just in case you accidentally add parentheses to an already-instantiated Pretokeniser object.
        return self


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

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        for pretokeniser in reversed(self.sequence):
            pretokens = pretokeniser.invertTokens(pretokens)
        return pretokens


class IdentityPretokeniser(Pretokeniser):

    def split(self, text: str) -> List[str]:
        return [text]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return pretokens


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

    def __init__(self, hyphen_mode: "PunctuationPretokeniser.HyphenMode", protect_apostrophes_without_spaces: bool=True,
                 group_adjacent_spaces_with_punctuation: SpaceMarkerLocation=SpaceMarkerLocation.TOKEN):
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
        punctuation_escaped_no_accent = punctuation_escaped.replace("'", "")

        if hyphen_mode == PunctuationPretokeniser.HyphenMode.ONLY:
            pattern = "[" + punctuation_escaped + "]+"
        else:
            if protect_apostrophes_without_spaces:
                if group_adjacent_spaces_with_punctuation == SpaceMarkerLocation.START:
                    # For space grouping + protected accents,
                    #   \s?[',]*[,]+[',]*\s?|(\s|^)'|'(\s|$)
                    punctuation_groups = r"\s?[" + punctuation_escaped + "]*[" + punctuation_escaped_no_accent + "]+[" + punctuation_escaped + "]*"
                    starting_accent = r"(\s|^)'"
                    ending_accent   = r"'(?=\s|$)"
                elif group_adjacent_spaces_with_punctuation == SpaceMarkerLocation.END:
                    punctuation_groups = "[" + punctuation_escaped + "]*[" + punctuation_escaped_no_accent + "]+[" + punctuation_escaped + r"]*\s?"
                    starting_accent = r"(?<=\s|^)'"
                    ending_accent   = r"'(\s|$)"
                else:
                    # For no space grouping + protected accents,
                    #   [',]*[,]+[',]*|(?<=\s|^)'|'(?=\s|$)
                    punctuation_groups = "[" + punctuation_escaped + "]*[" + punctuation_escaped_no_accent + "]+[" + punctuation_escaped + "]*"
                    starting_accent = r"(?<=\s|^)'"
                    ending_accent   = r"'(?=\s|$)"
                pattern = punctuation_groups + "|" + starting_accent + "|" + ending_accent
            else:
                # For space grouping + no protected accents,
                #   \s?[',]+\s?
                # For no space grouping + no protected accents,
                #   [',]+
                pattern = r"\s?" * (group_adjacent_spaces_with_punctuation == SpaceMarkerLocation.START) + \
                          "[" + punctuation_escaped + "]+" + \
                          r"\s?" * (group_adjacent_spaces_with_punctuation == SpaceMarkerLocation.END)

        self.core = tp.Split(pattern=Regex(pattern), behavior="isolated")

    def split(self, text: str) -> List[str]:
        return [w for w, _ in self.core.pre_tokenize_str(text)]

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Probably needs some kind of intelligent English punctuation rule.
        return pretokens


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
            return [] if not(self.marker.location == SpaceMarkerLocation.TOKEN and text) else [self.marker.substitute]

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
            chars, sow = WhitespaceAndMarkerPretokeniser.stripMarker(pretoken, self.marker)
            if self.marker.detached:
                return [sow] + list(chars)
            else:
                return [sow + chars[0]] + list(chars[1:])
        elif self.marker.location == SpaceMarkerLocation.END:
            chars, eow = WhitespaceAndMarkerPretokeniser.stripMarker(pretoken, self.marker)
            if self.marker.detached:
                return list(chars) + [eow]
            else:
                return list(chars[:-1]) + [chars[-1] + eow]
        elif self.marker.location == SpaceMarkerLocation.TOKEN:
            if pretoken == self.marker.substitute:
                return [pretoken]
            else:
                return list(pretoken)
        else:
            return [pretoken]

    @staticmethod
    def stripMarker(pretoken: str, marker: SpaceMarker) -> Tuple[str,str]:
        """
        Retrieve the part of a pretoken that isn't a space marker.
        """
        L = len(marker.substitute)
        if marker.location == SpaceMarkerLocation.START:
            root, mark = pretoken[L:], pretoken[:L]
            if mark != marker.substitute:
                root, mark = pretoken, ""
        elif marker.location == SpaceMarkerLocation.END:
            root, mark = pretoken[:len(pretoken)-L], pretoken[len(pretoken)-L:]
            if mark != marker.substitute:
                root, mark = pretoken, ""
        elif marker.location == SpaceMarkerLocation.TOKEN:
            if pretoken == marker.substitute:
                root, mark = "", pretoken
            else:
                root, mark = pretoken, ""
        else:
            root, mark = pretoken, ""

        return root, mark

    def invertToken(self, pretoken: str) -> str:
        return pretoken.replace(self.marker.substitute, " ")  # TODO: Technically should not do replacements in the middle.

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return [self.invertToken(p) for p in pretokens]

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
        """
        :param destructive: If false, whitespace is not removed, but just separated from other characters.
        """
        if destructive:
            self.pattern = re.compile(r"[\s​]+")
        else:
            self.pattern = re.compile(r"([\s​]+)")

    def split(self, text: str) -> List[str]:
        pretokens = self.pattern.split(text)
        return [t for t in pretokens if t]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return WhitespaceAndMarkerPretokeniser.intercalate(pretokens, " ")


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

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Should actually look for adjacent digits and merge them.
        return pretokens


class EnglishApostrophes(Pretokeniser):

    def __init__(self, do_nt=True):
        if do_nt:
            self.pattern = regex.compile(r"""('s|'re|'ve|'m|'ll|'d|n't)(?=\s|$|\p{P})""", re.IGNORECASE)
        else:  # This is the GPT-2 standard (except GPT-2 doesn't ignore case).
            self.pattern = regex.compile(r"""('s|'re|'ve|'m|'ll|'d|'t)(?=\s|$|\p{P})""", re.IGNORECASE)

    def split(self, text: str) -> List[str]:
        return [p for p in self.pattern.split(text) if p]

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Should be smarter.
        return pretokens


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

    def invertToken(self, pretoken: str) -> str:
        # return pretoken.replace(self.marker.substitute, " ")  # TODO: Technically should not do replacements in the middle.
        return WhitespaceAndMarkerPretokeniser.stripMarker(pretoken, self.marker)[0]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return [self.invertToken(p) for p in pretokens]


from .mappers import InvertibleTextMapper
class MapperAsPretokeniser(Pretokeniser):
    """
    When used in a sequence of pretokenisers, this allows you to apply a text->text mapping on individual pretokens
    produced by another pretokeniser. In HuggingFace, for example, you can't do this.
    """

    def __init__(self, mapper: InvertibleTextMapper):
        self.core = mapper

    def split(self, text: str) -> List[str]:
        return [self.core.convert(text)]

    def invertToken(self, pretoken: str) -> str:
        return self.core.invert(pretoken)

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return [self.invertToken(p) for p in pretokens]


class HuggingFacePretokeniser(Pretokeniser):

    def __init__(self, encoder: tp.PreTokenizer, decoder: td.Decoder):
        """
        Steals the pretokeniser from a HuggingFace tokeniser.
        Only possible for the "Fast" variants because some people don't know how to design a software system.
        https://github.com/huggingface/transformers/issues/26254
        """
        self.encode: tp.PreTokenizer = encoder
        self.decode: td.Decoder      = decoder

    def split(self, text: str) -> List[str]:
        return [w for w, _ in self.encode.pre_tokenize_str(text)]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return self.decode.decode(pretokens)

    @staticmethod
    def fromFullTokeniser(hf_model: PreTrainedTokenizerFast) -> "HuggingFacePretokeniser":
        return HuggingFacePretokeniser(hf_model.backend_tokenizer.pre_tokenizer, hf_model.backend_tokenizer.decoder)
