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

from ..preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from ..util.iterables import intercalate


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

    def getName(self):
        return self.__class__.__name__


class PretokeniserSequence(Pretokeniser):

    def __init__(self, sub_pretokenisers: List[Pretokeniser]):
        self.sequence = sub_pretokenisers

    def split(self, text: str) -> List[str]:
        current_pretokens = [text]
        # print(current_pretokens)
        for pretokeniser in self.sequence:
            generated_pretokens = []
            for pretoken in current_pretokens:
                generated_pretokens.extend(pretokeniser.split(pretoken))

            current_pretokens = generated_pretokens
            # print("\t->", pretokeniser.getName(), "->", current_pretokens)

        return current_pretokens

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        # print(pretokens)
        for pretokeniser in reversed(self.sequence):
            pretokens = pretokeniser.invertTokens(pretokens)
            # print("\t->", pretokeniser.getName() + "⁻¹", "->", pretokens)
        return pretokens


class IdentityPretokeniser(Pretokeniser):

    def split(self, text: str) -> List[str]:
        return [text]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return pretokens


class HyphenMode(Enum):
    ONLY     = 1
    EXCLUDED = 2
    INCLUDED = 3


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

    def __init__(self, hyphen_mode: HyphenMode=HyphenMode.INCLUDED, protect_apostrophes_without_spaces: bool=True,
                 group_adjacent_spaces_with_punctuation: BoundaryMarkerLocation=BoundaryMarkerLocation.ISOLATED):
        punctuation = PunctuationPretokeniser.buildPunctuationString(hyphen_mode)
        punctuation_escaped = punctuation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("-", "\\-")
        punctuation_escaped_no_accent = punctuation_escaped.replace("'", "")

        if hyphen_mode == HyphenMode.ONLY:
            pattern = "[" + punctuation_escaped + "]+"
        else:
            if protect_apostrophes_without_spaces:
                if group_adjacent_spaces_with_punctuation == BoundaryMarkerLocation.START:
                    # For space grouping + protected accents,
                    #   \s?[',]*[,]+[',]*\s?|(\s|^)'|'(\s|$)
                    punctuation_groups = r"\s?[" + punctuation_escaped + "]*[" + punctuation_escaped_no_accent + "]+[" + punctuation_escaped + "]*"
                    starting_accent = r"(\s|^)'"
                    ending_accent   = r"'(?=\s|$)"
                elif group_adjacent_spaces_with_punctuation == BoundaryMarkerLocation.END:
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
                pattern = r"\s?" * (group_adjacent_spaces_with_punctuation == BoundaryMarkerLocation.START) + \
                          "[" + punctuation_escaped + "]+" + \
                          r"\s?" * (group_adjacent_spaces_with_punctuation == BoundaryMarkerLocation.END)

        self.core = tp.Split(pattern=Regex(pattern), behavior="isolated")

    @staticmethod
    def buildPunctuationString(hyphen_mode: HyphenMode=HyphenMode.INCLUDED):
        punctuation_hyphens_only = "-–_"
        if hyphen_mode == HyphenMode.ONLY:
            punctuation = punctuation_hyphens_only
        else:
            punctuation = BASIC_PUNCTUATION + "€£…‘’“”„«»"      # Add some European punctuations.
            punctuation = punctuation.replace("\\", "") + "\\"  # Put backslash in the back. Makes the pattern clearer.
            for hyphen in punctuation_hyphens_only:
                punctuation = punctuation.replace(hyphen, "")  # Note: Not all of these will have effect, but some will.

            if hyphen_mode == HyphenMode.INCLUDED:
                for hyphen in punctuation_hyphens_only:
                    punctuation = hyphen + punctuation  # Note: All of these will have effect.

        return punctuation

    def split(self, text: str) -> List[str]:
        return [w for w, _ in self.core.pre_tokenize_str(text)]

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Probably needs some kind of intelligent English punctuation rule.
        return pretokens


class DistinctPunctuation(Pretokeniser):
    """
    Punctuation splitters group all successive punctuation together, but you likely don't want to mix punctuation marks
    together into one token. So make groups of equal punctuation out of full-punctuation tokens.
    """

    def __init__(self):
        self.punctuation = PunctuationPretokeniser.buildPunctuationString(HyphenMode.INCLUDED)
        self.punctuation_group = re.compile(" ?[" + self.punctuation.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").replace("-", "\\-") + "]+ ?")

    def split(self, text: str) -> List[str]:
        if not self.punctuation_group.match(text):
            return [text]
        else:
            current_char = ""
            pretokens = []
            for char in text:
                if char != current_char:
                    current_char = char
                    pretokens.append("")
                pretokens[-1] += char

            return pretokens

    def invertTokens(self, pretokens: List[str]) -> List[str]:
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

    def __init__(self, replacement: BoundaryMarker):
        self.marker = replacement

    def split(self, text: str) -> List[str]:
        if not text.strip():
            return [] if not(self.marker.location == BoundaryMarkerLocation.ISOLATED and text) else [self.marker.substitute]

        pretokens = [text]
        if self.marker.location == BoundaryMarkerLocation.ISOLATED:
            pretokens = text.split()  # Will strip all whitespace from both sides, then split on any span of whitespace.
            pretokens = list(intercalate(pretokens, self.marker.substitute))  # Will have length 2n-1.
            if text[0].isspace():
                pretokens.insert(0, self.marker.substitute)
            if text[-1].isspace():  # Due to the sanity check above, we know that this is not the same whitespace!
                pretokens.append(self.marker.substitute)

        elif self.marker.location == BoundaryMarkerLocation.START:
            pretokens = text.split()
            for i in range(len(pretokens)):
                if i != 0 or text[0].isspace():
                    pretokens[i] = self.marker.substitute + pretokens[i]

        elif self.marker.location == BoundaryMarkerLocation.END:
            pretokens = text.split()
            for i in range(len(pretokens)):
                if i != len(pretokens)-1 or text[-1].isspace():
                    pretokens[i] = pretokens[i] + self.marker.substitute

        return pretokens

    def invertToken(self, pretoken: str) -> str:
        return pretoken.replace(self.marker.substitute, " ")  # TODO: Technically should not do replacements in the middle.

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return [self.invertToken(p) for p in pretokens]


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
        return list(intercalate(pretokens, " "))


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

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Should actually look for adjacent digits and merge them. Careful merging sequences like "2024 10 02" to "20241002" though.
        return pretokens


class EnglishApostrophes(Pretokeniser):
    """
    Splits English contractions ('ve, 'll, 'd, 's, 're, ...) off the rest of the word.
    """

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

    The reason this is a pretokeniser is that it can produce multiple tokens at once (namely, a separate token as boundary).
    """

    def __init__(self, marker: BoundaryMarker):
        self.marker = marker

    def split(self, text: str) -> List[str]:
        if self.marker.location == BoundaryMarkerLocation.ISOLATED:
            return [self.marker.substitute, text]
        elif self.marker.location == BoundaryMarkerLocation.START:
            return [self.marker.substitute + text]
        elif self.marker.location == BoundaryMarkerLocation.END:
            return [text + self.marker.substitute]
        else:
            return [text]

    def invertToken(self, pretoken: str) -> str:
        return self.marker.isolate(pretoken)[0]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return [self.invertToken(p) for p in pretokens]

    def getName(self):
        return super().getName() + "(" + "+"*(self.marker.location == BoundaryMarkerLocation.END) + self.marker.substitute + "+"*(self.marker.location == BoundaryMarkerLocation.START) + ")"


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

    def getName(self):
        return super().getName() + "(" + self.core.__class__.__name__ + ")"


class InsertReverse(Pretokeniser):
    """
    Given tokens [a, b, c], inserts their reverses as [a, rev(a), b, rev(b), c, rev(c)].
    """

    def split(self, text: str) -> List[str]:
        return [text, text[::-1]]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        last_pretoken = ""
        preserved = []
        for pretoken in pretokens:
            if pretoken != last_pretoken[::-1]:  # If you aren't the reverse of the previous string, you need to be preserved.
                preserved.append(pretoken)
                last_pretoken = pretoken  # The next pretoken might be your reverse.
            else:
                last_pretoken = ""  # We have found the reverse, which means we must NOT include it nor compare it with the next pretoken.

        return preserved
