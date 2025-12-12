from dataclasses import dataclass

from ..interfaces.identifiers import Specials, NoSpecials, AutoVocab, AutoVocabSpecs


@dataclass
class BertSpecials(Specials):  # BERT actually has 994 [unusedXYZ] tokens, but I don't want to pollute this file with them. So if you really wanted to portray BERT accurately, you'd need an ExtendedBertSpecials or whatever.
    CLS: int
    SEP: int
    PAD: int
    MASK: int

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return [self.CLS] + ids + [self.SEP]

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return [self.CLS] + ids_1 + [self.SEP] + ids_2 + [self.SEP]


@dataclass
class RobertaSpecials(Specials):  # RoBERTa has "madeupword0000", "madeupword0001" and "madeupword0002" special tokens.
    BOS: int
    EOS: int
    PAD: int
    MASK: int

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return [self.BOS] + ids + [self.EOS]

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return [self.BOS] + ids_1 + [self.EOS, self.EOS] + ids_2 + [self.EOS]


@dataclass
class GptSpecials(Specials):  # GPT actually doesn't even have an UNK.
    PAD: int  # GPT actually does not have a separate padding token to EOS; it technically does not matter which token is used as pad because its definition is "that token which is masked out of all computations". It is however conceptually cleaner AND more importantly you absolutely need SpecialTokensMixin to have a pad_token for things to work properly.
    BOS: int  # <|endoftext|>

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return [self.BOS] + ids  # Yes, endoftext is the start of the text. https://github.com/karpathy/build-nanogpt/discussions/51

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return [self.BOS] + ids_1 + [self.BOS] + ids_2


@dataclass
class LLamaSpecials(GptSpecials):  # LLama's tokeniser by default uses the same template as GPT-2, but can be configured to do the below.
    EOS: int

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return [self.BOS] + ids + [self.EOS]

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return self._singleSentenceTemplate(ids_1) + self._singleSentenceTemplate(ids_2)


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

    # TODO: I have assumed OLMo uses the template of GPT-2 for its tokenisation, which ironically has an endoftext at the beginning but not the end.

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return [self.EOS] + ids

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return [self.EOS] + ids_1 + [self.EOS] + ids_2


@dataclass
class ModernBertSpecials(Specials):  # Should be extended with [UNUSED0]...[UNUSED82]
    bert: BertSpecials
    olmo: OLMoSpecials

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return self.bert._singleSentenceTemplate(ids)

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return self.bert._pairedSentenceTemplate(ids_1, ids_2)


@dataclass
class ChatMLSpecials(Specials):
    """
    ChatML (Chat Markup Language) is the conversation format used by OpenAI.
    https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md
    """
    MESSAGE_START: int  # <|im_start|>  the "im" is for "input message" https://community.openai.com/t/what-do-the-im-start-and-im-end-tokens-mean/145727
    MESSAGE_END: int    # <|im_end|>


@dataclass
class QwenVLVisualSpecials(Specials):
    """
    See section 2.2 in https://arxiv.org/pdf/2308.12966
    """
    # ViT embedding delimiters
    VISION_START: int  # Called <img> in Qwen-VL and <|vision_start|> in Qwen2-VL https://arxiv.org/pdf/2409.12191v1
    VISION_END: int

    # Description of a bounding box (also called an "object reference")
    CAPTION_START: int  # <ref> in Qwen-VL and <|object_ref_start|> in Qwen2-VL
    CAPTION_END: int    # </ref>

    # Bounding box coordinate delimiters, used for grounding tasks (given an image and an object description, generate coordinates)
    BBOX_START: int  # <box> in Qwen-VL and <|box_start|> in Qwen2-VL
    BBOX_END: int
    QUAD_START: int  # Starts 4 pairs of coordinates rather than 2.
    QUAD_END: int


@dataclass
class Qwen2VLVisualSpecials(QwenVLVisualSpecials):
    """
    The way Qwen2-VL processes images and videos is that rather than delimiting them with two pairs of tags
    (image start/end, video start/end), you delimit them both with the same vision start/end pair, and then in the
    middle you initially put a single placeholder which is substituted either by image embeddings or video embeddings.
    """
    IMAGE_PLACEHOLDER: int  # Also called <|image_pad|>
    VIDEO_PLACEHOLDER: int  # Also <|video_pad|>


@dataclass
class Qwen2VLSpecials(Specials):
    # If you look at Qwen3-VL's list of specials at https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/tokenizer_config.json
    # it actually has a few specials that have completely no purpose:
    #   <|vision_pad|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<think>", "</think>",
    # and then there are four tool-related tokens that aren't reported by HF's special_tokens_map
    #   <tool_response>", "</tool_response>", "<tool_call>", "</tool_call>
    # We don't include any of these.
    text: GptSpecials
    vision: Qwen2VLVisualSpecials
