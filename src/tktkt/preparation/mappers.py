from abc import ABC, abstractmethod
from typing import List

import requests

import tokenizers.normalizers as tn
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode


class TextMapper(ABC):
    @abstractmethod
    def convert(self, text: str) -> str:
        pass


class InvertibleTextMapper(TextMapper):
    @abstractmethod
    def invert(self, text: str) -> str:
        pass


#####################################################################


class MapperSequence(TextMapper):

    def __init__(self, submappers: List[TextMapper]):
        self.sequence = submappers

    def convert(self, text: str) -> str:
        for mapper in self.sequence:
            text = mapper.convert(text)
        return text


class HuggingFaceNormaliser(TextMapper):

    def __init__(self, core: tn.Normalizer):
        self.hf = core

    def convert(self, text: str) -> str:
        return self.hf.normalize_str(text)

    @staticmethod
    def fromFullTokeniser(hf_model: PreTrainedTokenizerFast) -> "HuggingFaceNormaliser":
        return HuggingFaceNormaliser(hf_model.backend_tokenizer.normalizer or tn.Sequence([]))


class Stripper(TextMapper):

    def convert(self, text: str) -> str:
        return text.strip()


class Lowercaser(TextMapper):

    def convert(self, text: str) -> str:
        return text.lower()


from .splitters import Pretokeniser
class DilatePretokens(TextMapper):
    """
    Ensure there are spaces everywhere a pretokeniser indicates there need to be.
    """

    def __init__(self, pretokeniser: Pretokeniser):
        self.p = pretokeniser

    def convert(self, text: str) -> str:
        return " ".join(self.p.split(text))


class AsPhonemes(TextMapper):

    def __init__(self, dictionary_language: str="eng-us"):
        import text2phonemesequence as TeetwoPiece
        self.model = TeetwoPiece.Text2PhonemeSequence(language="", pretrained_g2p_model='charsiu/g2p_multilingual_byT5_small_100', is_cuda=True)
        self._initialiseLanguageDictionary(language=dictionary_language)

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


#####################################################################


class InvertibleMapperSequence(InvertibleTextMapper):

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


class IdentityMapper(InvertibleMapperSequence):

    def __init__(self):
        super().__init__([])


class AppendSpace(InvertibleTextMapper):

    def __init__(self, front_not_back: bool):
        self.front = front_not_back

    def convert(self, text: str) -> str:
        return " "*self.front + text + " "*(not self.front)

    def invert(self, text: str) -> str:
        trim_front = text[0].isspace() and self.front
        trim_back  = text[-1].isspace() and not self.front
        return text[trim_front : len(text) - trim_back]


class Replace(InvertibleTextMapper):

    def __init__(self, old: str, new: str):
        self.old = old
        self.new = new

    def convert(self, text: str) -> str:
        return text.replace(self.old, self.new)

    def invert(self, text: str) -> str:
        return text.replace(self.new, self.old)


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


def bytes_to_unicode_documented():
    """
    An implementation that's actually readable.

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
    bytes_without_mapping = [byte for byte in range(2**8) if byte not in bytes_with_mapping]

    mappings = bytes_with_mapping[:]
    n = 0
    for byte in bytes_without_mapping:
        bytes_with_mapping.append(byte)
        mappings.append(2**8 + n)
        n += 1

    codepoints_of_mappings = [chr(n) for n in mappings]
    return dict(zip(bytes_with_mapping, codepoints_of_mappings))


def bytes_to_unicode_softcoded():
    """
    Same mapping except without any hardcoding.
    """
    offset = 0
    mappings = dict()
    for byte in range(256):
        representation = chr(byte)
        if representation.isspace() or len(representation.__repr__()) == 6:  # Cannot be printed properly in a text file.
            mappings[byte] = chr(256 + offset)
            offset += 1
        else:
            mappings[byte] = representation

    return mappings


BYTE_TO_PSEUDO = bytes_to_unicode()
PSEUDO_TO_BYTE = {v: k for k, v in BYTE_TO_PSEUDO.items()}
PSEUDO_TO_BYTE[" "] = PSEUDO_TO_BYTE["Ġ"]

SPACING_BYTES = [9, 10, 13, 32]

class PseudoByteMapping(InvertibleTextMapper):
    """
    Converts each character to its UTF-8 encoding's "pseudo-bytes", which are themselves characters that each represent
    a unique byte, even though they in turn don't necessarily have any relationship to that byte.
    """

    def convert(self, text: str) -> str:
        return "".join(map(BYTE_TO_PSEUDO.get, text.encode("utf-8")))

    def invert(self, text: str) -> str:
        return bytes(map(PSEUDO_TO_BYTE.get, text)).decode("utf-8", errors="replace")
