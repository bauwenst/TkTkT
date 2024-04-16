from typing import List, Mapping
from copy import deepcopy

from transformers import PreTrainedTokenizerFast
import tokenizers.pre_tokenizers as tp
import tokenizers.normalizers as tn

from ...interfaces.tokeniser import TokeniserWithVocab
from ...preparation.instances import HuggingFacePreprocessorForWords, HuggingFacePreprocessor


class HuggingFaceTokeniser(TokeniserWithVocab):
    """
    Takes a HuggingFace tokeniser and splits it into its pretokeniser and core tokeniser.
    This way, the user can choose whether to apply the pretokeniser or not.
    """

    def __init__(self, wrapped_tokeniser: PreTrainedTokenizerFast, for_single_words: bool=False):
        if not for_single_words:  # Copy whatever pretokeniser hangs onto the wrapped model.
            preprocessor = HuggingFacePreprocessor(wrapped_tokeniser)
        else:  # Do that, but add additional components that ensure that all input is interpreted as a word, regardless of spacing.
            preprocessor = HuggingFacePreprocessorForWords(wrapped_tokeniser)
        super().__init__(preprocessor)

        # Disable the wrapped tokeniser's preprocessing steps. This means that calling .tokenize() now ignores the pretokeniser.
        wrapped_tokeniser = deepcopy(wrapped_tokeniser)
        wrapped_tokeniser.backend_tokenizer.normalizer    = tn.Sequence([])
        wrapped_tokeniser.backend_tokenizer.pre_tokenizer = tp.Sequence([])
        self.backend = wrapped_tokeniser

    def tokenise(self, pretoken: str) -> List[str]:
        """
        Tokenises without pretokenisation.

        Note that for HuggingFace tokenisers that had a byte-based pretokeniser originally, it is still aware of the
        byte alphabet (possibly due to what's in the vocab) and DELETES every character it doesn't recognise.
        No UNKs, just delete. For such tokenisers, you have to ensure manually that you don't use out-of-alphabet characters.
        """
        return self.backend.tokenize(pretoken)

    def typeToId(self, t: str) -> int:
        return self.backend._convert_token_to_id(t)

    def idToType(self, i: int) -> str:
        return self.backend._convert_id_to_token(i)

    def getVocabMapping(self) -> Mapping[str, int]:
        return self.backend.get_vocab()

    def getVocabSize(self) -> int:
        return self.backend.vocab_size
