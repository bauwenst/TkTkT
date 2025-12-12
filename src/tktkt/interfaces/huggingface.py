"""
Interface for tokenisers that want to be compatible with the HuggingFace suite.

Indeed, this has nothing to do with having a tokeniser that runs on HuggingFace internally, because that would be just
another member of tktkt.models. Rather, this is about needs to be implemented in order to become a HuggingFace tokeniser.
"""
from typing import List, Optional, Tuple, Mapping, Iterable, Callable
from abc import ABC, abstractmethod
from pathlib import Path

import warnings
import json
from transformers import PreTrainedTokenizer, SpecialTokensMixin

from .tokeniser import TokeniserWithVocabulary, WithSpecials
from .identifiers import Specials


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

    @abstractmethod
    def get_special_tokens_mask(self, token_ids_0: list, token_ids_1: Optional[list]=None, already_has_special_tokens: bool=False) -> list[int]:
        pass


class TktktToHuggingFace(HuggingFaceTokeniserInterface):
    """
    Wrap any TkTkT tokeniser with a vocabulary so that it adheres to the above interface.

    We don't do this by default because then that TkTkT tokeniser's class would become polluted with HuggingFace shit
    which we want to avoid (the entire raison d'être of TkTkT...).
    """

    def __init__(self, backend: TokeniserWithVocabulary[WithSpecials], specials_map: dict[str,str]=None, specials_formatter: Callable[[str],str]=None, **kwargs):
        """
        :param specials_map: Maps the constructor arguments of SpecialTokensMixin, i.e. "bos_token", "eos_token", ...
                             to the special keys in the given tokeniser's Vocab's Specials.
                             For example: {"pad_token": "PAD", "eos_token": "EOS", ...}
        """
        self.backend = backend
        self._specials_formatter = specials_formatter or (lambda s: "[" + s + "]")

        if specials_map is None:
            # formatted_specials_map = AutoSpecials.fromTktkt(self.backend.vocab.specials, specials_formatter=self._specials_formatter).special_tokens_map  # TODO: deprecated; this method is now called from the Specials class by default.
            specials_map           = self.backend.vocab.specials._hfSpecialsMap()
            formatted_specials_map = {k: self._specials_formatter(v) for k,v in specials_map.items()}
        else:
            formatted_specials_map = {k: self._specials_formatter(v) for k,v in specials_map.items()}
        if self.backend.vocab.UNK is not None:
            formatted_specials_map["unk_token"] = self._specials_formatter("UNK")

        assert not(set(formatted_specials_map.values()) & set(self.backend.vocab))

        # HF allows you to either ADD them or DECLARE specials. I choose to declare them, because adding them isn't done safely. Note that HuggingFace doesn't check whether declared special tokens exist in the vocab (and will return the ID for UNK if you ask for their ID), but our Vocab does ensure this safety by construction.
        # Adding them:
        #   self.add_special_tokens(formatted_specials_map)  # We cannot use this because in case the tokens are missing, it makes new IDs using len(vocab) and that could be an existing ID after knockout.
        # Declaring them:
        kwargs.update(formatted_specials_map)
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return self.backend.vocab.size()

    def get_vocab(self) -> dict[str,int]:
        return self.backend.vocab.unsafe(specials_formatter=self._specials_formatter)

    def _convert_token_to_id(self, token: str) -> int:
        try:  # try-except is the fastest method to check+return a key in use cases where most lookups are valid (https://stackoverflow.com/a/28860508/9352077). Additionally, you don't pre-evaluate the unk ID, which otherwise causes an infinite loop since self.unk_token_id is actually a method call that itself calls _convert_token_to_id to get the ID of UNK (and assumes that this exists). Hence, you should only evaluate self.unk_token_id when you actually need it, not just any call.
            return self.backend.typeToId(token)
        except:
            return self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.backend.idToType(index)

    def _tokenize(self, text, **kwargs) -> list[str]:
        return self.backend.tokenise(text)  #, **kwargs)  # TODO: Supporting HF's kwargs isn't important, right?

    def tokenize(self, text: str, **kwargs) -> list[str]:
        """
        Overrides the implementation of .tokenize() in `transformers` altogether, because it is basically a selective
        pretokeniser on top of ._tokenize(), and we don't want to allow the possibility of selectively not tokenising
        the surface string belonging to a special in the unsafe vocabulary.
        The tokeniser itself cannot do this itself because it doesn't use the unsafe vocabulary, so it will always
        segment those surface strings IF GIVEN them.
        """
        return self.backend.prepareAndTokenise(text)  #, **kwargs)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.backend.preprocessor.undo(tokens)

    ### Boilerplate below

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> tuple[str]:
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

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]=None) -> list[int]:
        if token_ids_1 is None:
            return self.backend.vocab.specials._singleSentenceTemplate(token_ids_0)
        else:
            return self.backend.vocab.specials._pairedSentenceTemplate(token_ids_0, token_ids_1)

    def get_special_tokens_mask(self, token_ids_0: list, token_ids_1: Optional[list]=None, already_has_special_tokens: bool=False) -> list[int]:
        if already_has_special_tokens:
            assert token_ids_1 is None  # If not, you're claiming you got specials added to paired sequences BEFORE combining them, which is impossible.

            special_id_set = set(self.backend.vocab.specials)
            return [int(id in special_id_set) for id in token_ids_0]
        else:
            ids_with_specials = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
            return self.get_special_tokens_mask(ids_with_specials, already_has_special_tokens=True)


class AutoSpecials:
    """
    Turns arguments into HuggingFace SpecialTokensMixin objects.
    """

    @staticmethod
    def fromStrings(types: Iterable[str]) -> SpecialTokensMixin:
        """
        Attempt to find all the special types in the given domain (e.g. [CLS], [SEP], etc...).
        Delimiters cannot be mixed (e.g. not both [CLS] and </s>, or [UNK] and <mask>, etc...).
        """
        DELIMITERS = {0: ("[", "]"), 1: ("<", ">")}

        def warnIfAlreadyAssigned(should_be_none, value) -> bool:
            if should_be_none is None:
                return True
            else:
                warnings.warn(f"Value '{value}' was not assigned to a variable since it already contained the value '{should_be_none}'.")
                return False

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

    @staticmethod
    def fromTktkt(specials: Specials, specials_formatter: Callable[[str], str]=None) -> SpecialTokensMixin:
        """
        Tries to automatically map the given TkTkT specials to a HuggingFace mixin.

        Of course, you could (and should!) do this manually if you know the subclass the Specials object belongs to.
        """
        if specials_formatter is None:
            specials_formatter = lambda s: "[" + s + "]"

        def warnIfAlreadyAssigned(should_be_none, value) -> bool:
            if should_be_none is None:
                return True
            else:
                warnings.warn(f"Value '{value}' was not assigned to a variable since it already contained the value '{should_be_none}'.")
                return False

        mixin = SpecialTokensMixin()

        for special, _ in specials.__iter_keys__():
            special_lower = special.lower()
            special = specials_formatter(special)
            # id = getattr_recursive(specials, special)  # As it turns out, HF's SpecialTokensMixin does NOT store identifiers. It stores strings that are passed to the token-to-id convertor method. Horrific.

            if "bos" in special_lower:
                if warnIfAlreadyAssigned(mixin.bos_token, special):
                    mixin.bos_token = special
            elif "eos" in special_lower or "endoftext" in special_lower:
                if warnIfAlreadyAssigned(mixin.eos_token, special):
                    mixin.eos_token = special
            elif "cls" in special_lower:
                if warnIfAlreadyAssigned(mixin.cls_token, special):
                    mixin.cls_token = special
            elif "sep" in special_lower:
                if warnIfAlreadyAssigned(mixin.sep_token, special):
                    mixin.sep_token = special
            elif "pad" in special_lower:
                if warnIfAlreadyAssigned(mixin.pad_token, special):
                    mixin.pad_token = special
            elif "msk" in special_lower or "mask" in special_lower:
                if warnIfAlreadyAssigned(mixin.mask_token, special):
                    mixin.mask_token = special

        return mixin
