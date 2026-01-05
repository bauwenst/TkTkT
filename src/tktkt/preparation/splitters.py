"""
Pretokenisation, i.e. splitting text into the units that will be tokenised separately.
Most of the class names use a prefix to indicate what they do:
    - "On" means you're looking for substrings to remove and split whenever you do.
    - "Into" means every string you produce is conceptually the same.
    - "Isolate" means you look for strings to separate from their left and right context, but keep intact.
    - "Group" means you isolate despite your neighbouring context looking similar.
"""
from typing import List, Optional, Union
from enum import Enum
import numpy as np

from tokenizers import Regex
from tokenizers import pre_tokenizers as tp
from string import punctuation as BASIC_PUNCTUATION
import re
import regex  # Has \p{} classes

from .boundaries import BoundaryMarker, BoundaryMarkerLocation
from ..interfaces.preprocessors import Pretokeniser, InvertibleTextMapper, _PreprocessorComponentSequence
from ..interfaces.tokenisers import Tokeniser
from ..util.iterables import intercalate
from ..util.strings import maskToTokens


class PretokeniserSequence(Pretokeniser, _PreprocessorComponentSequence):

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

    def _diagnosticName(self) -> str:
        return "Sequence(" + "+".join([p._diagnosticName() for p in self.sequence]) + ")" if self.sequence else "Identity"

    def __iter__(self):
        for pretokeniser in self.sequence:
            yield from iter(pretokeniser)


class PretokeniserSequence_Diagnostic(PretokeniserSequence):
    """
    Identical to a PretokeniserSequence except it prints every step it executes and the results from doing so.
    This is not just a flag in the PretokeniserSequence constructor because that would cause a lot of overhead
    on the critical path of all preprocessors (either an empty method call or a conditional statement).
    """

    def __init__(self, sub_pretokenisers: List[Pretokeniser]):
        super().__init__(sub_pretokenisers)
        max_name_length =          max(map(len, (p._diagnosticName()  for p in sub_pretokenisers)))
        self._indents   = [max_name_length - len(p._diagnosticName()) for p in sub_pretokenisers]

    def split(self, text: str) -> List[str]:
        current_pretokens = [text]
        print(current_pretokens)
        for pretokeniser, indent in zip(self.sequence, self._indents):
            generated_pretokens = []
            for pretoken in current_pretokens:
                generated_pretokens.extend(pretokeniser.split(pretoken))

            current_pretokens = generated_pretokens
            print("\t->", pretokeniser._diagnosticName() + " " * indent, "->", current_pretokens)

        return current_pretokens

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        print(pretokens)
        for pretokeniser, indent in zip(reversed(self.sequence), reversed(self._indents)):
            pretokens = pretokeniser.invertTokens(pretokens)
            print("\t->", pretokeniser._diagnosticName() + "⁻¹" + " " * indent, "->", pretokens)
        return pretokens


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


class IsolatePunctuation(Pretokeniser):
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
        punctuation = IsolatePunctuation.buildPunctuationString(hyphen_mode)
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
    def buildPunctuationString(hyphen_mode: HyphenMode=HyphenMode.INCLUDED):  # TODO: You might consider \p{P} from the regex package.
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


class GroupDistinctPunctuation(Pretokeniser):
    """
    Make groups of equal punctuation out of full-punctuation tokens.
    Punctuation splitters group all successive punctuation together, but you likely don't want to mix punctuation marks
    together into one token. For example:

        "This sentence ends in a lot of punctuation (that is different)...!"

    has a punctuation string ")...!" at the end, which should be split into ")" and "..." and "!".
    """

    def __init__(self):
        self.punctuation = IsolatePunctuation.buildPunctuationString(HyphenMode.INCLUDED)
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


class OnWhitespaceAndAddMarker(Pretokeniser):
    """
    Splits on (and DESTROYS) consecutive whitespace, replacing it by a marker.

    You should probably not use this class; WhitespacePretokeniser and AddWordBoundary will do the job.

    A general principle I adhere to is that the addition of space markings should NOT be signalled by the user text.
    For example, the user should not have to put a space in front of his input to make the tokeniser put a marker
    in front of the input.
    If the user wants to control, for the same tokeniser, if a marker is added, it should be an argument to
    the .prepareAndTokenise() method.
    """

    def __init__(self, replacement: BoundaryMarker):
        self.marker = replacement

    def _modifyAlphabet(self, known: list[str]) -> list[str]:
        return known + [self.marker.substitute]

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


class OnRegex(Pretokeniser):
    """
    Uses the given regular expression to identify a separator to split on.
    """
    def __init__(self, separator_pattern: Union[re.Pattern,regex.Pattern], destructive: bool=True):
        """
        :param destructive: If False, the separator is not removed, but just separated from other characters.
        """
        is_group = separator_pattern.pattern[0] == "(" and separator_pattern.pattern[-1] == ")"
        if destructive and is_group:  # This is contradictory.
            while is_group:
                separator_pattern = regex.compile(separator_pattern.pattern[1:-1])
                is_group = separator_pattern.pattern[0] == "(" and separator_pattern.pattern[-1] == ")"
        elif not destructive and not is_group:  # This is also contradictory.
            separator_pattern = regex.compile("(" + separator_pattern.pattern + ")")
        self.pattern = separator_pattern

    def split(self, text: str) -> List[str]:
        return [t for t in self.pattern.split(text) if t]

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # Should be overridden if possible.
        return pretokens


class IntoRegexGroups(Pretokeniser):
    """
    Uses the given regular expression to identify pretokens. Everything that doesn't match is discarded.
    """
    def __init__(self, pretoken_pattern: Union[re.Pattern,regex.Pattern]):
        self.pattern = pretoken_pattern

    def split(self, text: str) -> List[str]:
        return self.pattern.findall(text)

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # Actually not invertible.
        return pretokens


class OnWhitespace(OnRegex):
    """
    Isolates whitespace and (optionally) destroys it.

    Slightly broader splitter than Python's native .split() since this also reacts to word separators that aren't
    visible to the naked eye, e.g. ZWSP.
    """

    def __init__(self, destructive: bool=True):
        """
        :param destructive: If False, whitespace is not removed, but just separated from other characters.
        """
        self._keep_spaces = not destructive
        super().__init__(separator_pattern=re.compile(r"[\s​]+"), destructive=destructive)

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: This is actually very wrong. If you start out with > 1 pretoken, and only one of them (or neither) is split on a space, then inverting by putting a space between EVERY pretoken is clearly wrong.
        if self._keep_spaces:
            return pretokens
        else:
            return list(intercalate(pretokens, " "))

WhitespacePretokeniser = OnWhitespace


class IntoWhitespacePrefixed(IntoRegexGroups):
    """
    Rather than isolating/deleting whitespace, this pretokeniser finds whitespace and splits right before or right after it,
    including the whitespace in the resulting tokens with non-whitespace characters.

    The start/end of a string counts as whitespace.
    """

    def __init__(self, prefix_not_suffix: bool=True):
        if prefix_not_suffix:
            pattern = re.compile(r"(?:^|\s|​)[^\s​]*")
        else:
            pattern = re.compile(r"[^\s​]+(?:$|\s|​)")  # The + is intentional, although I can't really explain why it works.
        self._before_not_after = prefix_not_suffix
        super().__init__(pretoken_pattern=pattern)

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


class IntoSentences(Pretokeniser):
    """
    Uses the 3-layer XLM-R model from "Segment any Text (SaT)" to find sentence boundaries.
    https://arxiv.org/abs/2406.16678
    """

    def __init__(self, cuda: bool=True):
        from wtpsplit import SaT
        self._backend = SaT("sat-3l-sm")
        if cuda:
            self._backend.half().to("cuda")

    def split(self, text: str) -> List[str]:
        return self._backend.split(text)

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return ["".join(pretokens)]  # We're assuming this is basically the first thing being run. Note that spaces are included in SaT.


class IsolateNumbers(OnRegex):
    r"""
    Explanation of the regex:
    (?:             The following group, without capturing it.
        \p{N}       A character that counts as a number
        |           or
        [.,]        a comma or point
        (?=\p{N})   which sees ahead of it another character that counts as a number.
    )+              And of this group, one or more.
    """
    def __init__(self):
        super().__init__(separator_pattern=regex.compile(r"(?:\p{N}|[.,](?=\p{N}))+"), destructive=False)


class IsolateDigits(OnRegex):
    """
    Isolate all digits into single-character tokens. If you don't know why we need this, read
    https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/

    Works for any language and any numerical character. Try e.g. ૪ (4 in Gujarati) or ௲ (1000 in Tamil) or ½.
    Reference: https://character.construction/numbers
    """

    def __init__(self):
        super().__init__(separator_pattern=regex.compile(r"\p{N}"), destructive=False)

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Should actually look for adjacent digits and merge them. Careful merging sequences like "2024 10 02" to "20241002" though.
        return pretokens


class GroupDigits(Pretokeniser):
    """
    Groups digits into sequences of at most N digits, working right-to-left. For example, for N=3, the number
        1234567
    would be pretokenised into
        1 234 567
    rather than
        123 456 7
    """
    def __init__(self, n: int=3):
        self.pattern_to_get_runs_of_numbers = regex.compile(r"(\d+)")
        self.pattern_to_split_numbers       = regex.compile(r"(\d{" + str(n) + r"})(?=\d)")

    def split(self, text: str) -> List[str]:
        pretokens = []

        is_digit = True
        for run in self.pattern_to_get_runs_of_numbers.split(text):
            is_digit = not is_digit
            if is_digit:
                pretokens.extend(self.pattern_to_split_numbers.sub(r"\1/", run[::-1])[::-1].split("/"))
            elif run:
                pretokens.append(run)

        return pretokens

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # TODO: Again, you could do some work to concatenate digits.
        return pretokens


class IsolateConnectingHyphens(OnRegex):
    """
    Isolates hyphens that are surrounded by non-space on both sides.
    """

    def __init__(self):
        pattern = re.compile(r"((?<!-)(?<=\S)-+(?=\S)(?!-))")  # Maximally large spans of hyphens, but only take those that do have a character on both sides that is non-space, but not those cases where those characters are just hyphens.
        super().__init__(pattern, destructive=False)

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        new_pretokens = []
        buffer = []
        active_hyphen = False
        for pretoken in pretokens:
            if all(c == "-" for c in pretoken) and buffer:
                active_hyphen = True
                buffer.append(pretoken)
            elif active_hyphen:
                buffer.append(pretoken)
                active_hyphen = False
            else:
                if buffer:
                    new_pretokens.append("".join(buffer))
                    buffer = []
                buffer.append(pretoken)
        if buffer:
            if len(buffer) > 1 and all(c == "-" for c in buffer[-1]):
                new_pretokens.append("".join(buffer[:-1]))
                buffer = [buffer[-1]]
            new_pretokens.append("".join(buffer))

        return new_pretokens


class PolariseApostrophes(Pretokeniser):
    """
    Generalised version of the EnglishContractions pretokeniser.

    Splits on apostrophes and concatenates them to their shortest neighbour.
    Thus, "pa'que" will be split "pa' que" but "they'll" will be split "they 'll".
    Additionally, two consecutive apostrophes will always be separated.
    """

    def __init__(self, tiebreak_left: bool):
        self._left = tiebreak_left

    def split(self, text: str) -> List[str]:
        pretokens = text.split("'")
        lengths = [len(pretoken) for pretoken in pretokens]
        for i in range(len(lengths)-1):
            if lengths[i+1] == 0 or (lengths[i] != 0 and (lengths[i] < lengths[i+1] or (self._left and lengths[i] == lengths[i+1]))):  # This condition has the nice property that it never concatenates two apostrophes together and always sticks them to characters where available. E.g.: if you were to write I'''m, it would become I' ' 'm. Also, just 'm stays 'm.
                pretokens[i] += "'"
            else:
                pretokens[i+1] = "'" + pretokens[i+1]

        return [pretoken for pretoken in pretokens if pretoken]

    def invertTokens(self, pretokens: List[str]) -> List[str]:  # You can't really know whether an apostrophe was already split off before this, and thus whether it should stay split off after.
        return pretokens


class IsolateEnglishContractions(OnRegex):
    """
    Splits English contractions ('ve, 'll, 'd, 's, 're, ...) off the rest of the word.
    """

    def __init__(self, do_nt=True):
        if do_nt:
            pattern = regex.compile(r"""('s|'re|'ve|'m|'ll|'d|n't)(?=\s|$|\p{P})""", re.IGNORECASE)
        else:  # This is the GPT-2 standard (except GPT-2 doesn't ignore case).
            pattern = regex.compile(r"""('s|'re|'ve|'m|'ll|'d|'t)(?=\s|$|\p{P})""", re.IGNORECASE)
        super().__init__(pattern, destructive=False)

EnglishApostrophes = IsolateEnglishContractions


class IntoJapaneseWords(Pretokeniser):

    def __init__(self):
        from fugashi import GenericTagger as MecabWrapper
        import ipadic  # IPAdic as dictionary makes MeCab behave like a word tokeniser. UniDic would make it behave like a morphological analyser.
        self.backend = MecabWrapper(ipadic.MECAB_ARGS + ' -Owakati')

    def split(self, text: str) -> List[str]:
        nodes = self.backend.parseToNodeList(text)  # Produces a list of elements which have type  from fugashi import Node
        return [node.surface for node in nodes]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        return pretokens


class IntoThaiWords(Pretokeniser):

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

    def _modifyAlphabet(self, known: list[str]) -> list[str]:
        return known + [self.marker.substitute]

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
        """
        Because we know that this pretokeniser adds exactly one word boundary, a sequence with multiple pretokens
        can be treated according to the rule that if a pretoken doesn't end with a boundary and the next pretoken
        doesn't start with a boundary, they can be safely concatenated.

        Similar implementations are found in DistinctPunctuation and SplitNextToWhitespace.
        """
        buffer = []
        new_pretokens = []
        for pretoken in pretokens:
            root, mark = self.marker.isolate(pretoken)
            if mark:
                if self.marker.location == BoundaryMarkerLocation.END:  # The next pretoken will start a new run.
                    buffer.append(root)

                if buffer:
                    new_pretokens.append("".join(buffer))
                buffer = []

                if self.marker.location == BoundaryMarkerLocation.START:  # You have found the start of a new run.
                    buffer.append(root)
            else:
                buffer.append(root)

        if buffer:
            new_pretokens.append("".join(buffer))

        return new_pretokens

    def _diagnosticName(self):
        return super()._diagnosticName() + "(" + "+"*(self.marker.location == BoundaryMarkerLocation.END) + self.marker.substitute + "+"*(self.marker.location == BoundaryMarkerLocation.START) + ")"


class AddCapitalMarker(Pretokeniser):
    """
    Lowercases characters, but adds special pretokens in front to losslessly recover the capitals later.

    Note: this class does not work after a byte mapping has been applied.
    """
    # TODO: "delete-space" character
    # TODO: It is quite possible that these special characters "⇧" have to be protected somehow so that users themselves
    #       cannot inject them (this used to be avoided by universal alphabets, but they have fallen into disfavour).

    def __init__(self, ignore_marker: BoundaryMarker=None):
        """
        :param ignore_marker: If the incoming text is known to carry a boundary marker, you should ignore it when checking case.
        """
        self.marker = ignore_marker if ignore_marker is not None else BoundaryMarker("", detached=False, location=BoundaryMarkerLocation.START)

    def _modifyAlphabet(self, known: list[str]) -> list[str]:
        return known + ["⇧", "⇪"]

    def split(self, text: str) -> List[str]:
        root, mark = self.marker.isolate(text)
        if not root:
            return [mark] if mark else []
        elif root.isupper():  # Caps-lock
            return ["⇪", self.marker.concatenate(root.lower(), mark)]
        elif root[0].isupper():  # Capitalised
            return ["⇧", self.marker.concatenate(root[0].lower() + root[1:], mark)]
        else:
            return [text]

    def invertTokens(self, pretokens: List[str]) -> List[str]:
        shift     = False
        caps_lock = False
        recased_pretokens = []
        for pretoken in pretokens:
            if pretoken == "⇪":
                caps_lock = True
                shift     = False
            elif pretoken == "⇧":
                caps_lock = False
                shift     = True
            elif caps_lock:
                root, mark = self.marker.isolate(pretoken)
                recased_pretokens.append(self.marker.concatenate(root.upper(), mark))
                caps_lock = False
            elif shift:
                root, mark = self.marker.isolate(pretoken)
                recased_pretokens.append(self.marker.concatenate(root[0].upper() + root[1:], mark))
                shift = False
            else:
                recased_pretokens.append(pretoken)

        return recased_pretokens



from ..models.predictive.viterbi import CharacterClassifier
class OnBoundaryClassifier(Pretokeniser):

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

    def _diagnosticName(self):
        return super()._diagnosticName() + "(" + self.core.__class__.__name__ + ")"


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
