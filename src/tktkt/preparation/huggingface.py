from typing import List

from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
import tokenizers.normalizers as tn
import tokenizers.pre_tokenizers as tp
import tokenizers.decoders as td

from .mappers import MapperSequence, Stripper, AppendSpace, TextMapper, Pretokeniser
from .boundaries import BoundaryMarkerLocation
from .splitters import PretokeniserSequence, BoundaryMarker, MapperAsPretokeniser
from ..interfaces.preparation import Preprocessor


class HuggingFaceNormaliser(TextMapper):

    def __init__(self, core: tn.Normalizer):
        self.hf = core

    def convert(self, text: str) -> str:
        return self.hf.normalize_str(text)

    @staticmethod
    def fromFullTokeniser(hf_model: PreTrainedTokenizerFast) -> "HuggingFaceNormaliser":
        if hf_model.backend_tokenizer.normalizer is None:  # For some reason, bool(hf_model.backend_tokenizer.normalizer) == False and yet it isn't None!
            return HuggingFaceNormaliser(tn.Sequence([]))
        else:
            return HuggingFaceNormaliser(hf_model.backend_tokenizer.normalizer)


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


class HuggingFacePreprocessor(Preprocessor):

    def __init__(self, hf_model: PreTrainedTokenizerFast):
        super().__init__(
            uninvertible_mapping=HuggingFaceNormaliser.fromFullTokeniser(hf_model),
            splitter=HuggingFacePretokeniser.fromFullTokeniser(hf_model)
        )


class HuggingFacePreprocessorForWords(Preprocessor):
    """
    For tokenising text as if it was an isolated word (i.e. the start of the word is the start of the input and the end
    is the end of the input). Not trivial since by default, tokenisers like the RobertaTokenizerFast assume a string is
    explicitly not at the start of a word if there is no start-of-word marker.

    The motivation behind this implementation:
        - In our framework, adding markers like a start-of-word is straight-forward: you add them to everything after
          splitting on spaces and punctuation, with the idea being that a word should always be started with a SoW (or
          ended with an EoW) regardless of the spaces that surround it. You don't encode text, you encode semantics.
        - We don't, however, have control over the SoW/EoW behaviour of HuggingFace tokenisers, and they DO make
          use of spaces to decide whether to put down a SoW, we are forced to add a space to the input.
        - Since we don't want the user to have to change their input depending on which tokeniser they use, we give them
          this preprocessor to wrap their tokeniser in and hence it will behave consistently.
    """

    def __init__(self, hf_model: PreTrainedTokenizerFast):
        super().__init__(
            uninvertible_mapping=MapperSequence([
                HuggingFaceNormaliser.fromFullTokeniser(hf_model),
                Stripper()  # Whatever the HF normaliser does, we want to control all space.
            ]),
            splitter=PretokeniserSequence([
                MapperAsPretokeniser(AppendSpace(front_not_back=True)),  # We know the HF pretokeniser uses spaces as word boundary, so we add it first.
                HuggingFacePretokeniser.fromFullTokeniser(hf_model)
            ])
        )


def detectByteBased(hf_tokeniser: PreTrainedTokenizerBase) -> bool:
    tests = [
        ("號", "èĻŁ")
    ]

    for test, charset in tests:
        tokens = hf_tokeniser.tokenize(test)
        charset = set(charset)
        for pretoken in tokens:
            charset -= set(pretoken)
            if not charset:  # Passed the test
                break
        else:  # Failed the test
            return False

    return True


def detectBoundaryMarker(hf_tokeniser: PreTrainedTokenizerBase) -> BoundaryMarker:
    """
    Assumes a couple of things about the marker:
        - It will always appear attached to another letter, even though we assume it starts out as "detached" during tokenisation.
        - It only appears once for very long strings, i.e. it is not a continuation symbol. This isn't an issue for BERT-style
          tokenisers because there, the tokeniser itself actually adds the continuation symbol.
    """
    CHAR = "a"
    N = 50

    token_with_potential_prefix = hf_tokeniser.tokenize(" " + CHAR*N)[0]
    if CHAR in token_with_potential_prefix and token_with_potential_prefix.rstrip(CHAR) and token_with_potential_prefix != token_with_potential_prefix.rstrip(CHAR):
        prefix = token_with_potential_prefix.rstrip(CHAR)
        return BoundaryMarker(prefix, detached=True, location=BoundaryMarkerLocation.START)

    token_with_potential_suffix = hf_tokeniser.tokenize(CHAR*N + " ")[-1]
    if CHAR in token_with_potential_suffix and token_with_potential_suffix.lstrip(CHAR) and token_with_potential_suffix != token_with_potential_suffix.lstrip(CHAR):
        suffix = token_with_potential_suffix.lstrip(CHAR)
        return BoundaryMarker(suffix, detached=True, location=BoundaryMarkerLocation.END)

    # continuation = hf_tokeniser.tokenize("a"*100)[1].rstrip("a")  # TODO: Does TkTkT even support BERT continuation?
    # print("P:", prefix)
    # print("S:", suffix)
    return BoundaryMarker("", detached=True, location=BoundaryMarkerLocation.START)
