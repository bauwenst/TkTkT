"""
Interfaces to make tokenisers that are compatible with the HuggingFace suite.
Indeed, these are interfaces that TkTkT tokenisers hold to (like all the other TkTkT interfaces),
in this case with the goal of becoming HuggingFace-compatible.
"""
from typing import List, Optional, Tuple, Mapping, Dict, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

import warnings
import json
from transformers import PreTrainedTokenizer, SpecialTokensMixin

from .tokeniser import TokeniserWithFiniteTypeDomain


class HuggingFaceTokeniserInterface(PreTrainedTokenizer, ABC):
    """
    The base class for a Pythonic HuggingFace tokeniser (transformers.PreTrainedTokenizer) has unimplemented methods,
    yet it does NOT tag these with @abstractmethod and instead just assumes you will notice when they throw a
    NotImplementedError.

    This class explicitly defines them as abstract methods to force inheritors to implement these methods if they
    want to be instantiated. Note that it is NOT a Protocol, because (1) a Protocol doesn't inherit implementations
    and (2) it is not us wanting to add a third party's classes to our hierarchy, but the reverse, so we can just use their existing base class.
    """

    @abstractmethod
    def _tokenize(self, text, **kwargs) -> List[str]:
        pass

    # The following three methods are for interfacing with the vocabulary.

    @property  # Property because that's how HuggingFace does it. Makes no sense to have getter/setter for this, but ok.
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def _convert_token_to_id(self, token: str) -> int:
        pass

    @abstractmethod
    def _convert_id_to_token(self, index: int) -> str:
        pass

    # The following two methods are for storage and come from the parent class of PreTrainedTokenizer.

    @abstractmethod
    def get_vocab(self) -> Mapping[str,int]:
        pass

    @abstractmethod
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        pass

    # The following methods are technically already implemented in HF, but it's important to define them explicitly.

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        pass

    @abstractmethod
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        """
        Takes over the role of tokenizers.processors (adding [CLS] and [SEP]) in PreTrainedTokenizer, where it is called by:
            ._encode_plus()
                .tokenize(text)
                    ._tokenize(text)
                .convert_tokens_to_ids(tokens)
                .prepare_for_model(ids)
                    .build_inputs_with_special_tokens(ids)
                    .create_token_type_ids_from_sequences(...)
        """
        pass


class TktktToHuggingFace(HuggingFaceTokeniserInterface):
    """
    Wrap any TkTkT tokeniser with a vocabulary so that it adheres to the above interface.

    We don't do this by default because then that TkTkT tokeniser's class would become polluted with HuggingFace shit
    which we want to avoid (the entire raison d'être of TkTkT...).
    """

    def __init__(self, backend: TokeniserWithFiniteTypeDomain, specials_from: SpecialTokensMixin=None, **kwargs):
        self.backend = backend

        if specials_from is None:
            specials_from = detectSpecials(self.backend.types())

        # TODO: Since this constructor is given the specials that we want, you can actually quite safely add them to
        #       the vocab if it's a dictionary. You don't let HF do it because HF can't be trusted with making new IDs, but we can.
        # Special tokens: HF allows you to either ADD them or DECLARE them. I choose to declare them, because adding them
        #                 isn't done safely. Also, because HuggingFace doesn't check whether declared special tokens exist
        #                 in the vocab (and will return the ID for UNK if you ask for their ID), I do that here.
        vocab_keys = set(self.backend.types())                           # key "[UNK]"     -> value 0
        special_values = set(specials_from.special_tokens_map.values())  # key "unk_token" -> value "[UNK]"
        assert len(vocab_keys - special_values) == len(vocab_keys) - len(special_values)
        # Adding them:
        #   self.add_special_tokens(specials_from.special_tokens_map)  # We cannot use this because in case the tokens are missing, it makes new IDs using len(vocab) and that could be an existing ID after knockout.
        # Declaring them:
        kwargs.update(specials_from.special_tokens_map)
        super().__init__(**kwargs)

    def _tokenize(self, text, **kwargs) -> List[str]:
        return self.backend.tokenise(text, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self.backend.getVocabSize()

    def get_vocab(self) -> Mapping[str,int]:
        return {t: self.backend.typeToId(t) for t in self.backend.types()}

    def _convert_token_to_id(self, token: str) -> int:
        try:  # try-except is the fastest method to check+return a key in use cases where most lookups are valid (https://stackoverflow.com/a/28860508/9352077). Additionally, you don't pre-evaluate the unk ID, which otherwise causes an infinite loop since self.unk_token_id is actually a method call that itself calls _convert_token_to_id to get the ID of UNK (and assumes that this exists). Hence, you should only evaluate self.unk_token_id when you actually need it, not just any call.
            return self.backend.typeToId(token)
        except:
            return self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.backend.idToType(index)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return self.backend.prepareAndTokenise(text, **kwargs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.backend.preprocessor.undo(tokens)

    ### Boilerplate below

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        mapping = self.get_vocab()
        if not isinstance(mapping, dict):
            warnings.warn("Vocabulary was not saved because it isn't a dictionary.")
            return ()
        else:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)

            filename_prefix = "" if filename_prefix is None else filename_prefix + "_"
            file_path = save_directory / (filename_prefix + "vocab.json")  # Will be overwritten if it already exists. Tough luck!
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(mapping, handle)

            return (file_path.as_posix(),)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]


def detectSpecials(types: Iterable[str]) -> SpecialTokensMixin:
    """
    Attempt to find all the special types in the given domain (e.g. [CLS], [SEP], etc...).
    Delimiters cannot be mixed (e.g. not both [CLS] and </s>, or [UNK] and <mask>, etc...).
    """
    def warnIfAlreadyAssigned(should_be_none, value) -> bool:
        if should_be_none is None:
            return True
        else:
            warnings.warn(f"Value '{value}' was not assigned to a variable since it already contained the value '{should_be_none}'.")
            return False

    DELIMITERS = {0: ("[", "]"), 1: ("<", ">")}

    special_families = {i: [] for i in DELIMITERS}
    for t in types:
        for i, (left, right) in DELIMITERS.items():
            if t[0] == left and t[-1] == right:
                special_families[i].append(t)

    i, found_types = sorted(special_families.items(), key=lambda t: len(t[1]), reverse=True)[0]
    mixin = SpecialTokensMixin()
    for t in found_types:
        t_lower = t.lower()
        if "bos" in t_lower:
            if warnIfAlreadyAssigned(mixin.bos_token, t):
                mixin.bos_token = t
        elif "eos" in t_lower:
            if warnIfAlreadyAssigned(mixin.eos_token, t):
                mixin.eos_token = t
        elif "cls" in t_lower:
            if warnIfAlreadyAssigned(mixin.cls_token, t):
                mixin.cls_token = t
        elif "sep" in t_lower:
            if warnIfAlreadyAssigned(mixin.sep_token, t):
                mixin.sep_token = t
        elif "pad" in t_lower:
            if warnIfAlreadyAssigned(mixin.pad_token, t):
                mixin.pad_token = t
        elif "unk" in t_lower:
            if warnIfAlreadyAssigned(mixin.unk_token, t):
                mixin.unk_token = t
        elif "msk" in t_lower or "mask" in t_lower:
            if warnIfAlreadyAssigned(mixin.mask_token, t):
                mixin.mask_token = t
        elif "/s" in t_lower:
            if warnIfAlreadyAssigned(mixin.eos_token, t):
                mixin.eos_token = t
        elif "s" in t_lower:
            if warnIfAlreadyAssigned(mixin.bos_token, t):
                mixin.bos_token = t
        elif "<|endoftext|>" in t_lower:
            if warnIfAlreadyAssigned(mixin.eos_token, t):
                mixin.eos_token = t
        else:
            warnings.warn(f"Found special-seeming but unrecognisable type: {t}")

    return mixin
