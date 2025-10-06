from typing import List, Iterable, Dict, Union, Set, Optional
from collections import OrderedDict

import re
import regex
import requests
import unicodedata

from .boundaries import BoundaryMarker
from ..interfaces.preparation import TextMapper, InvertibleTextMapper, Pretokeniser, FiniteCharacterSet, _PreprocessorComponentSequence
from ..util.dicts import invertdict, insertKeyAlias


class MapperSequence(TextMapper, _PreprocessorComponentSequence):

    def __init__(self, submappers: List[TextMapper]):
        self.sequence = submappers

    def convert(self, text: str) -> str:
        for mapper in self.sequence:
            text = mapper.convert(text)
        return text

    def __iter__(self):
        for mapper in self.sequence:
            yield from iter(mapper)


class Stripper(TextMapper):
    """
    Strips whitespace off both ends of a given string.
    """
    def convert(self, text: str) -> str:
        return text.strip()


class Lowercaser(TextMapper):
    """
    Lowercases a given string.
    """
    def convert(self, text: str) -> str:
        return text.lower()


class Truncate(TextMapper):
    """
    Truncates a given string to a maximal length in characters.
    Can help curb the danger of tokenising massive examples.
    """
    def __init__(self, maximum_character_count: int):
        self.cap = maximum_character_count

    def convert(self, text: str) -> str:
        return text[:self.cap]


class TruncateOnNearestWhitespace(TextMapper):
    """
    Same as Truncate(), except you go looking for the nearest whitespace to the left or right of the limit.
    If you can't find any within a certain patience, this is entirely equivalent to Truncate().
    """
    def __init__(self, desired_character_limit: int, patience: int=50, forwards_search: bool=False):
        """
        :param patience: How many characters at most the space can be removed from the character at the desired limit.
                         For example, patience=1 allows finding a space before/after the last character that would be included.
        """
        last_normal_index = desired_character_limit-1  # Index of the last character we want included.
        if forwards_search:
            self._lower = last_normal_index             # Index of the first character to detect whitespace at.
            self._upper = last_normal_index+patience+1  # 1+index of the last character to detect whitespace at.
        else:
            self._lower = max(0, last_normal_index - patience)
            self._upper = last_normal_index+1

        self._largest_normal_length = desired_character_limit
        self._forwards = forwards_search

    def convert(self, text: str) -> str:
        if len(text) <= self._largest_normal_length:
            return text

        try:  # Try-except because we expect most cases to find a space.
            if self._forwards:
                return text[:text.index(" ", self._lower, self._upper)]
            else:
                return text[:text.rindex(" ", self._lower, self._upper)]
        except:
            return text[:self._largest_normal_length]


class FilterRegex(TextMapper):
    """
    Erases everything that matches the given pattern.
    """

    def __init__(self, pattern: Union[re.Pattern, regex.Pattern], replacement: str=""):
        self.pattern = pattern
        self.replacement = replacement

    def convert(self, text: str) -> str:
        return self.pattern.sub(self.replacement, text)


class FilterCharacters(FilterRegex):
    def __init__(self, charset: Iterable[str]):
        super().__init__(re.compile("[" + re.escape(charset) + "]"))


class FilterWhitespace(FilterRegex):
    def __init__(self):
        super().__init__(re.compile(r"\s"))


class FilterHyperlinks(FilterRegex):  # Source: https://stackoverflow.com/a/6041965/9352077
    def __init__(self):
        super().__init__(re.compile(r"""(https?|ftp)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])"""))


class SwallowIfContains(TextMapper):
    """
    Given an input, returns the empty string if any substring matches the given regular expression, else just returns
    the same input. This is not the same as simply filtering out the part of the string that matches.
    """

    def __init__(self, pattern: Union[re.Pattern, regex.Pattern]):
        self.pattern = pattern

    def convert(self, text: str) -> str:
        if self.pattern.search(text):
            return ""
        else:
            return text


class LimitRepetitions(TextMapper):
    """
    Reduce characters that repeat more than N times consecutively to N times instead.
    """

    def __init__(self, n: int):
        self.n = n

    def convert(self, text: str) -> str:
        new_text = []  # Yes, a list of characters, not a string.  https://stackoverflow.com/a/3055541/9352077

        prev_char = ""
        repeats = 0
        for c in text:
            if c != prev_char:
                prev_char = c
                repeats = 1
            else:
                repeats += 1

            if repeats <= self.n:
                new_text.append(c)

        return "".join(new_text)


class DilatePretokens(TextMapper):
    """
    Ensure there are spaces everywhere a pretokeniser indicates there need to be.
    """

    def __init__(self, pretokeniser: Pretokeniser):
        self.p = pretokeniser

    def convert(self, text: str) -> str:
        return " ".join(self.p.split(text))


class AsPhonemes(FiniteCharacterSet):

    def __init__(self, dictionary_language: str="eng-us"):
        import text2phonemesequence as TeetwoPiece
        self.model = TeetwoPiece.Text2PhonemeSequence(language="", pretrained_g2p_model='charsiu/g2p_multilingual_byT5_small_100', is_cuda=True)
        self._initialiseLanguageDictionary(language=dictionary_language)

    def getCharacters(self) -> Set[str]:  # FIXME: I actually have no clue where to even find the IPA. Best I can find is https://github.com/rhasspy/gruut-ipa which has _VOWELS and _CONSONANTS, but I don't know if that's enough.
        raise NotImplementedError

    def convert(self, text: str) -> str:
        return self.model.infer_sentence(text)

    def _initialiseLanguageDictionary(self, language: str):
        """
        The library uses wget to a file in the CWD. Not only does this command not work on Windows, but it is also just
        not pleasant to cache inside the CWD. Hence this method.
        """
        self.model.language = language
        if language + ".tsv" not in self.model.phoneme_length:
            return

        print(f"Retrieving phoneme dictionary for {language}...")
        url = "https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/" + language + ".tsv"
        response = requests.get(url)
        for word_phone in response.text.strip().split("\n"):
            w_p = word_phone.split("\t")
            assert len(w_p) == 2
            word,phone_string = w_p
            if "," not in phone_string:
                self.model.phone_dict[word] = [phone_string]
            else:
                self.model.phone_dict[word] = [phone_string.split(',')[0]]


import nlpaug
class PerturbWithNLPAug(TextMapper):
    """
    Wraps any object from the NLPaug package. See the possibilities with the following imports:
        import nlpaug.augmenter.char as nac
        import nlpaug.augmenter.word as naw
        import nlpaug.augmenter.sentence as nas
    """

    def __init__(self, nlp_aug: nlpaug.Augmenter):
        self._augmenter = nlp_aug

    def convert(self, text: str) -> str:
        return self._augmenter.augment(text)


#####################################################################


class InvertibleMapperSequence(InvertibleTextMapper, _PreprocessorComponentSequence):

    def __init__(self, submappers: List[InvertibleTextMapper]):
        self.sequence = submappers

    def convert(self, text: str) -> str:
        for mapper in self.sequence:
            text = mapper.convert(text)
        return text

    def invert(self, text: str) -> str:
        for mapper in reversed(self.sequence):
            text = mapper.invert(text)
        return text

    def __iter__(self):
        for mapper in self.sequence:
            yield from iter(mapper)


class IdentityMapper(InvertibleTextMapper):  # Not equivalent to InvertibleMapperSequence([]) because the latter doesn't return itself during __iter__ (which could be seen as a bug).

    def convert(self, text: str) -> str:
        return text

    def invert(self, text: str) -> str:
        return text


class AppendSpace(InvertibleTextMapper):

    def __init__(self, front_not_back: bool):
        self.front = front_not_back

    def convert(self, text: str) -> str:
        return " "*self.front + text + " "*(not self.front)

    def invert(self, text: str) -> str:
        trim_front = self.front     and len(text) and text[0].isspace()
        trim_back  = not self.front and len(text) and text[-1].isspace()
        return text[trim_front:len(text) - trim_back]


class Replace(InvertibleTextMapper):

    def __init__(self, old: str, new: str):
        if old == "":
            raise ValueError("Empty strings cannot be replaced because they cannot be found inside other strings.")
        if new == "":
            raise ValueError(f"Cannot replace string '{old}' by an empty string, since that is non-invertible. Use one of the Filter... mappers instead.")

        self.old = old
        self.new = new

    def convert(self, text: str) -> str:
        return text.replace(self.old, self.new)

    def invert(self, text: str) -> str:
        return text.replace(self.new, self.old)


class ReplaceBoundary(InvertibleTextMapper):
    """
    Replace, but only at the start or the end of the string.
    """

    def __init__(self, old: str, marker: BoundaryMarker):
        self.old = BoundaryMarker(substitute=old, detached=marker.detached, location=marker.location)
        self.new = marker

    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        return self.new

    def convert(self, text: str) -> str:
        root, marker_found = self.old.isolate(text)
        if marker_found:
            return self.new.concatenate(root, self.new.substitute)
        else:
            return text

    def invert(self, text: str) -> str:
        root, marker_found = self.new.isolate(text)
        if marker_found:
            return self.old.concatenate(root, self.old.substitute)
        else:
            return text


class MorphoChallengeCapitals(InvertibleMapperSequence):
    """
    The MorphoChallenge (http://morpho.aalto.fi/events/morphochallenge/) defines a character mapping for Turkish text
    where special characters are replaced by similar capitals. This is that mapping.
    """
    def __init__(self):
        super().__init__([
            Replace("ç", "C"),
            Replace("ı", "I"),
            Replace("ö", "O"),
            Replace("ü", "U"),
            Replace("ş", "S"),
            Replace("ğ", "G")
        ])


class ByteMapping(InvertibleTextMapper, FiniteCharacterSet):
    """
    Converts each character to its UTF-8 encoding's "pseudo-bytes", which are themselves characters that each represent
    a unique byte, even though they in turn don't necessarily have any relationship to that byte.
    """

    BYTE_TO_PSEUDO: Dict[int,str] = dict()
    PSEUDO_TO_BYTE: Dict[str,int] = dict()
    SPACING_BYTES = [9, 10, 13, 32]

    def convert(self, text: str) -> str:
        return "".join(map(self.BYTE_TO_PSEUDO.get, text.encode("utf-8")))

    def invert(self, text: str) -> str:
        try:
            return bytes(map(self.PSEUDO_TO_BYTE.get, text)).decode("utf-8", errors="replace")
        except:
            raise ValueError(f"Text contains one or more non-pseudobytes: {text}")


class PseudoByteMapping(ByteMapping):
    """
    Re-implements GPT-2's pseudo-byte mapping found at `transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode`.
    """

    @staticmethod
    def bytes_to_unicode_documented() -> Dict[int, str]:
        r"""
        An implementation of `transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode` that's actually readable.
        Has been tested for equivalence.

        The basic idea: most bytes are mapped to their codepoint when decoded with UTF-8, with the exception of the 68 bytes
        that have no representation.
            - The first 33 codepoints are unrepresentable bytes **and whitespace**. Both are difficult to represent in merges.txt.
            - The end of the 7-bit ASCII space is also a bunch of unrepresentable bytes.
            - There is a single unrepresentable byte after that (byte 173).

        The way these bytes and whitespace characters are mapped is simply by taking the first unused codepoint above byte 255.
        Note that it doesn't matter that the latter codepoints cannot themselves be encoded with 1 byte. All that matters is
        that they are one CHARACTER.

        The bytes that have a representable UTF-8 codepoint:
            First range (94):  !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
            Second range (12): ¡¢£¤¥¦§¨©ª«¬
            Third range (82):  ®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ

        There is a gap of 33 bytes before the first range, 34 between the first two, 1 between the last two.
        That makes 68 bytes not mapped in total.

        The first range of missing bytes contains tab (\t), newline (\n), carriage return (\r), and space.
        """
        bytes_with_mapping = list(range(ord("!"), ord("~") + 1)) \
                             + list(range(ord("¡"), ord("¬") + 1)) \
                             + list(range(ord("®"), ord("ÿ") + 1))
        bytes_without_mapping = [byte for byte in range(2 ** 8) if byte not in bytes_with_mapping]

        mappings = bytes_with_mapping[:]
        n = 0
        for byte in bytes_without_mapping:
            bytes_with_mapping.append(byte)
            mappings.append(2 ** 8 + n)
            n += 1

        codepoints_of_mappings = [chr(n) for n in mappings]
        return dict(zip(bytes_with_mapping, codepoints_of_mappings))

    @staticmethod
    def bytes_to_unicode_softcoded() -> Dict[int, str]:
        """
        Same mapping except without any hardcoding.
        """
        offset = 0
        mappings = OrderedDict()
        for byte in range(256):
            representation = chr(byte)
            if representation.isspace() or len(representation.__repr__()) == 6:  # Cannot be printed properly in a text file.
                representation = chr(256 + offset)
                offset += 1

            mappings[byte] = representation

        return mappings

    def getCharacters(self) -> List[str]:
        return list(PseudoByteMapping.bytes_to_unicode_softcoded().values())

    BYTE_TO_PSEUDO = bytes_to_unicode_softcoded()
    PSEUDO_TO_BYTE = insertKeyAlias(invertdict(BYTE_TO_PSEUDO), existing_key="Ġ", alias_key=" ")  # Map spaces (which technically shouldn't be in encoded input) to the byte that G maps to.


class LatinPseudoByteMapping(ByteMapping):
    """
    Like PseudoByteMapping, except for every pseudo-byte character that is NOT a Latin letter, you check:

        "Should the thing represented by this pseudo-byte (that thing being either an ASCII character or a UTF-8
         control byte) be groupable with any other byte?"

    When the answer is yes, then this character that is not a Latin letter should be remapped to a Latin letter.
        - If it's a UTF-8 control byte, the answer is always yes. This is true for
          bytes 161-172, bytes 174-191, byte 215, and byte 247.
        - If it's an ASCII character, the answer is no for two classes of characters:
            - Punctuation, EXCEPT for apostrophes, hyphens and underscores, which have some use for being grouped with letters if the user preprocesses their text such to make that available.
            - Bytes 0-32 and byte 127, which are control symbols. Caveat: these have pseudo-byte characters that already
              are Latin letters, and we don't change this. We only make groupalbe characters that became ungroupable, groupable again. Not the reverse.
    """

    @staticmethod
    def bytes_to_unicode() -> Dict[int,str]:
        LATINISED_PUNCTUATIONS = {"-", "'", "_"}

        mapping = PseudoByteMapping.bytes_to_unicode_softcoded()
        remaps = dict()
        offset = max(map(ord, mapping.values())) - 256 + 1  # offset value that hasn't been used yet.
        for byte,pseudo in mapping.items():
            pseudo_cat = unicodedata.category(pseudo)
            try:
                original_cat = unicodedata.category(bytes([byte]).decode("utf-8"))
            except:
                original_cat = ""

            # Remap if you're punctuation that isn't originally punctuation, or if it is, when it is requested that you do.
            if pseudo_cat == "Lo" or pseudo_cat == "No" or (
                    (pseudo_cat.startswith("S") or pseudo_cat.startswith("P")) and (
                            not(original_cat.startswith("P") or original_cat.startswith("S")) or
                            pseudo in LATINISED_PUNCTUATIONS
                    )
            ) or pseudo == "µ":  # µ is called "MICRO SIGN" and is a special exception.
                while not unicodedata.category(chr(256 + offset)).startswith("L"):  # Technically the L range has some bad characters, but not in the 128.
                    offset += 1
                remaps[byte] = chr(256 + offset)
                offset += 1

        return mapping | remaps

    def getCharacters(self) -> List[str]:
        return list(LatinPseudoByteMapping.bytes_to_unicode().values())

    BYTE_TO_PSEUDO = bytes_to_unicode()
    PSEUDO_TO_BYTE = insertKeyAlias(invertdict(BYTE_TO_PSEUDO), existing_key="Ġ", alias_key=" ")  # Map spaces (which technically shouldn't be in encoded input) to the byte that G maps to.
