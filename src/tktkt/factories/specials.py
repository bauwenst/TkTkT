from dataclasses import dataclass

from ..interfaces.identifiers import Specials, NoSpecials


@dataclass
class BertSpecials(Specials):  # BERT actually has 994 [unusedXYZ] tokens, but I don't want to pollute this file with them. So if you really wanted to portray BERT accurately, you'd need an ExtendedBertSpecials or whatever.
    CLS: int
    SEP: int
    PAD: int
    MASK: int


@dataclass
class RobertaSpecials(Specials):  # RoBERTa has "madeupword0000", "madeupword0001" and "madeupword0002" special tokens.
    BOS: int
    EOS: int
    PAD: int
    MASK: int


@dataclass
class GptSpecials(Specials):  # GPT actually doesn't even have an UNK.
    BOS: int  # <|endoftext|>


@dataclass
class PersonalIdentifiableInformationSpecials:  # PII anonymisation, which first appeared in the OLMo paper and was copied by ModernBERT.
    EMAIL: int
    PHONE: int
    IP_ADDRESS: int


@dataclass
class OLMoSpecials(Specials):
    EOS: int   # <|endoftext|>
    PAD2: int  # <|padding|>
    anonymisation: PersonalIdentifiableInformationSpecials


@dataclass
class ModernBertSpecials:  # Should be extended with [UNUSED0]...[UNUSED82]
    bert: BertSpecials
    olmo: OLMoSpecials


@dataclass
class Qwen3Specials(Specials):
    ENDOFTEXT: int
    IMAGE_START: int
    IMAGE_END: int
    IMAGE_PAD: int
    VISION_START: int
    VISION_END: int
    VISION_PAD: int
    VIDEO_PAD: int
    OBJECT_REFERENCE_START: int
    OBJECT_REFERENCE_END: int
    BOX_START: int
    BOX_END: int
    QUAD_START: int
    QUAD_END: int
