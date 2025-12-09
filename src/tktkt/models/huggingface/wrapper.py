from typing import Iterable, Union
from copy import deepcopy

from transformers import PreTrainedTokenizerFast
import tokenizers.pre_tokenizers as tp
import tokenizers.normalizers as tn

from ...interfaces.tokeniser import *
from ...interfaces.identifiers import AutoVocab, AutoVocabSpecs
from ...factories.preprocessing import HuggingFacePreprocessorForWords, HuggingFacePreprocessor


class HuggingFaceTokeniser(TokeniserWithVocabulary[WithSpecials]):
    """
    Takes a HuggingFace tokeniser and splits it into its pretokeniser and core tokeniser.
    This way, the user can choose whether to apply the pretokeniser or not.

    Note that all HuggingFace tokenisers have a known type-id bijection. This is even true for CANINE's tokeniser,
    which is just a UnicodeTokeniser. The catch is that it hashes the produced IDs inside the *model* rather than
    inside the *tokeniser* for doing lookups (and in fact, it uses sharded embeddings, i.e. one ID maps to multiple
    small embeddings based on different hash functions and then those are concatenated together as if it was looked up).
    """

    def __init__(self, wrapped_tokeniser: PreTrainedTokenizerFast, vocab_metadata: Union[Vocab[WithSpecials],AutoVocabSpecs[WithSpecials]], for_single_words: bool=False):
        if not for_single_words:  # Copy whatever pretokeniser hangs onto the wrapped model.
            preprocessor = HuggingFacePreprocessor(wrapped_tokeniser)
        else:  # Do that, but add additional components that ensure that all input is interpreted as a word, regardless of spacing.
            preprocessor = HuggingFacePreprocessorForWords(wrapped_tokeniser)

        if isinstance(vocab_metadata, AutoVocabSpecs):
            super().__init__(preprocessor=preprocessor, vocab=AutoVocab.fromTokenizer(wrapped_tokeniser, vocab_metadata))
        elif isinstance(vocab_metadata, Vocab):
            # Do a check to see that this Vocab at the very least doesn't violate what we know about the tokeniser.
            # Check 1: Does UNK match?
            unsafe_vocab = wrapped_tokeniser.get_vocab()
            special_map = wrapped_tokeniser.special_tokens_map
            actual_unk = special_map.pop("unk_token", None)
            assert actual_unk is None or unsafe_vocab[actual_unk] == vocab_metadata.UNK

            # Check 2: Are the known special IDs part of the given Vocab's specials at least?
            known_special_ids = set(vocab_metadata.specials)
            assert set(unsafe_vocab[t] for t in special_map.values()).issubset(known_special_ids)

            # Check 3: Do all remaining (type,id) pairs appear in the unsafe vocab?
            assert set(vocab_metadata.items()).issubset(set((t,i) for t,i in unsafe_vocab.items() if i not in known_special_ids))

            super().__init__(preprocessor=preprocessor, vocab=vocab_metadata)
        else:
            raise TypeError(type(vocab_metadata))

        # Disable the wrapped tokeniser's preprocessing steps. This means that calling .tokenize() now ignores the pretokeniser.
        wrapped_tokeniser = deepcopy(wrapped_tokeniser)
        wrapped_tokeniser.backend_tokenizer.normalizer    = tn.Sequence([])
        wrapped_tokeniser.backend_tokenizer.pre_tokenizer = tp.Sequence([])
        self.backend = wrapped_tokeniser

    def tokenise(self, pretoken: str) -> Tokens:
        """
        Tokenises without pretokenisation.

        Note that for HuggingFace tokenisers that had a byte-based pretokeniser originally, it is still aware of the
        byte alphabet (possibly due to what's in the vocab) and DELETES every character it doesn't recognise.
        No UNKs, just delete. For such tokenisers, you have to ensure manually that you don't use out-of-alphabet characters.
        """
        return self.backend.tokenize(pretoken)

    # TODO: All the below are probably bad approximations of the methods they override,
    #       which use Vocab's separation of types and specials, and we get that Vocab
    #       from AutoVocab which knows how to deal with HF's unk_token.

    def typeToId(self, t: str) -> int:
        return self.backend._convert_token_to_id_with_added_voc(t)

    def types(self) -> Iterable[str]:
        return self.backend.get_vocab().keys()

    def hasType(self, t: str) -> bool:
        return t in self.vocab

    def idToType(self, i: int) -> str:
        return self.backend._convert_id_to_token(i)

    def ids(self) -> Iterable[int]:
        return self.backend.get_vocab().values()

    def hasId(self, i: int) -> bool:
        return self.idToType(i) is not None  # self.backend.unk_token != self.idToType(i) doesn't work because HuggingFace is freaky like that.
