"""
Pretokenisation, i.e. splitting text into the units that will be tokenised separately.
"""
from typing import List, Optional
from enum import Enum
import numpy as np

from tokenizers import Regex
from tokenizers import pre_tokenizers as tp
from string import punctuation as BASIC_PUNCTUATION
import re
import regex  # Has \p{} classes

from .boundaries import BoundaryMarker, BoundaryMarkerLocation
from ..interfaces.preparation import Pretokeniser, InvertibleTextMapper, _PreprocessorComponentSequence
from ..interfaces.tokeniser import Tokeniser
from ..util.iterables import intercalate
from ..util.strings import maskToTokens


class PretokeniserSequence(Pretokeniser, _PreprocessorComponentSequence):

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

    def getName(self) -> str:
        return "Sequence(" + "+".join([p.getName() for p in self.sequence]) + ")" if self.sequence else "Identity"

    def __iter__(self):
        for pretokeniser in self.sequence:
            yield from iter(pretokeniser)


class IdentityPretokeniser(Pretokeniser):

    def split(self, text: str) -> List[str]:
        return [text]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return pretokens


class TokeniserAsPretokeniser(Pretokeniser):

    def __init__(self, tokeniser: Tokeniser, do_preprocess: bool=False):
        self._tokeniser = tokeniser
        self._do_preprocess = do_preprocess

    def split(self, text: str) -> List[str]:
        if self._do_preprocess:
            return self._tokeniser.prepareAndTokenise(text)
        else:
            return self._tokeniser.tokenise(text)


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
                 group_adjacent_spaces_with_punctuation: Optional[BoundaryMarkerLocation]=None):
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

    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        return self.marker

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
        self._keep_spaces = not destructive
        if destructive:
            self.pattern = re.compile(r"[\s​]+")
        else:
            self.pattern = re.compile(r"([\s​]+)")

    def split(self, text: str) -> List[str]:
        pretokens = self.pattern.split(text)
        return [t for t in pretokens if t]

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: This is actually very wrong. If you start out with > 1 pretoken, and only one of them (or neither) is split on a space, then inverting by putting a space between EVERY pretoken is clearly wrong.
        if self._keep_spaces:
            return pretokens
        else:
            return list(intercalate(pretokens, " "))


class SplitNextToWhitespace(Pretokeniser):

    def __init__(self, before_not_after: bool=True):
        if before_not_after:
            self.pattern = re.compile(r"(?:^|\s|​)[^\s​]*")
        else:
            self.pattern = re.compile(r"[^\s​]+(?:$|\s|​)")  # The + is intentional, although I can't really explain why it works.
        self._before_not_after = before_not_after

    def split(self, text: str) -> List[str]:
        return self.pattern.findall(text)

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        buffer = []
        new_pretokens = []
        if self._before_not_after:
            for pretoken in pretokens:
                if not pretoken[0].isspace() and buffer:  # => This is not a boundary that was split on originally, so flush the buffer.
                    new_pretokens.append("".join(buffer))
                    buffer = []
                buffer.append(pretoken)
            if buffer:
                new_pretokens.append("".join(buffer))
        else:
            for pretoken in reversed(pretokens):
                if not pretoken[-1].isspace() and buffer:
                    new_pretokens.insert(0, "".join(buffer))
                    buffer = []
                buffer.insert(0, pretoken)
            if buffer:
                new_pretokens.insert(0, "".join(buffer))

        return new_pretokens


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


class JapaneseWords(Pretokeniser):

    def __init__(self):
        from fugashi import GenericTagger as MecabWrapper
        import ipadic  # IPAdic as dictionary makes MeCab behave like a word tokeniser. UniDic would make it behave like a morphological analyser.
        self.backend = MecabWrapper(ipadic.MECAB_ARGS + ' -Owakati')

    def split(self, text: str) -> List[str]:
        nodes = self.backend.parseToNodeList(text)  # Produces a list of elements which have type  from fugashi import Node
        return [node.surface for node in nodes]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return pretokens


class ThaiWords(Pretokeniser):

    def __init__(self):
        from pythainlp.tokenize import Tokenizer as ThaiWordTokenizer
        self.backend = ThaiWordTokenizer(engine="newmm", keep_whitespace=False, join_broken_num=True)

    def split(self, text: str) -> List[str]:
        return self.backend.word_tokenize(text)

    def invertTokens(self, pretokens: List[str]) -> List[str]:
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

    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        return self.marker

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


from ..models.predictive.viterbi import CharacterClassifier
class SplitWithBoundaryClassifier(Pretokeniser):

    def __init__(self, classifier: CharacterClassifier, threshold: float=0.5):
        """
        :param classifier: Assigns a probability to each character for whether that is where the end of the pretoken is.
        :param threshold: Probabilities that are at least this will become pretoken boundaries.
        """
        self._classifier = classifier
        self._threshold = threshold

    def split(self, text: str) -> List[str]:
        mask = np.exp(self._classifier.getPointLogProbabilities(text)) >= self._threshold
        return maskToTokens(text, mask[:-1])

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return pretokens


class MapperAsPretokeniser(Pretokeniser, _PreprocessorComponentSequence):
    """
    When used in a sequence of pretokenisers, this allows you to apply a text->text mapping on individual pretokens
    produced by another pretokeniser. In HuggingFace, for example, you can't do this.
    """

    def __init__(self, mapper: InvertibleTextMapper):
        self.core = mapper

    def __iter__(self):
        yield from iter(self.core)

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
