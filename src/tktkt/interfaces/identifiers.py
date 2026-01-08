"""
Main objects involved in identifier (ID) mappings, particularly Specials and the Vocab.
"""
from typing import Iterator, Iterable, TypeVar, Generic, Callable, Optional, Union
from dataclasses import dataclass, is_dataclass, fields

from transformers import PreTrainedTokenizerBase

import warnings
from copy import deepcopy

from ..util.iterables import areContiguous, fst, areUnique, arePositive, snd
from ..util.dicts import getattr_recursive, setattr_recursive, intersect_dicts
from ..util.exceptions import EmptyTokenError


class _ProhibitDeclaringConstructor(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if namespace.get("__init__", None) is not None:  # namespace is basically what is written in code, but as a dictionary.
            raise AssertionError("Specials should have no __init__ constructor. They should be constructed through the @dataclass decorator.")


class _ProhibitSubclassing(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        for base in bases:
            if isinstance(base, _ProhibitSubclassing):
                raise TypeError(f"Illegal subclass '{name}' of parent class '{bases[0].__name__}'.")


class Specials(metaclass=_ProhibitDeclaringConstructor):  # This metaclass prevents the user from writing 'def __init__(self)' which means they are more or less forced to use a dataclass.
    """
    A possibly nested dataclass where each field stores the identifiers for a "special", i.e. an embedding in a
    language model that can only be added by the backend for control purposes, not through user input.

    Specials MUST NOT have a string representation. They can have an informal name like you and I just have a name
    (like "the emdding we use for classification"), but that name must never be represented as a string, and if it is,
    then the logic that converts it to an identifier must never be used by user input.

    HuggingFace used to agree, now they do it wrong. https://github.com/huggingface/transformers/issues/429
    And yes, the good old Bobby Tables joke applies. https://nicholas.carlini.com/writing/2023/little-bobby-endoftext.html
    """

    def __post_init__(self):  # This method is only called by @dataclass objects, namely at the end of their implicitly declared constructor. It is inherited, even if the base class is not a dataclass.
        assert is_dataclass(self)
        for field in fields(self):
            if field.type == int:
                value = getattr(self, field.name)
                if isinstance(value, bool) or not isinstance(value, int):  # You can't just ask "not isinstance(value, int)" because bools are ints lmao
                    raise TypeError(f"Special {field.name} was assigned a value of non-int type {type(value)}.")
                # if value not in {-1, +1}:
                #     raise ValueError(f"Special {field.name} should be initialised with a value of -1 or +1 (got {value}).")
            elif issubclass(field.type, Specials):
                pass
            else:
                raise TypeError(field.type)

        def getNames(specials: Specials) -> list[str]:
            all_names = list()
            for field in fields(specials):
                if issubclass(field.type, Specials):
                    nested_names = getNames(getattr(specials, field.name))
                    overlapping_names = set(nested_names) & set(all_names)
                    if len(overlapping_names) > 0:
                        raise ValueError(f"Special name(s) already taken: {list(overlapping_names)}")
                    all_names.extend(nested_names)
                elif field.type == int:
                    if field.name in all_names:
                        raise ValueError(f"Special name already taken: {field.name}")
                    all_names.append(field.name)
                else:
                    raise TypeError(field.type)
            return all_names

        names = getNames(self)
        assert "UNK" not in names, "The name 'UNK' is always included in the vocabulary and cannot be added as a special."

    def __iter__(self) -> Iterator[int]:
        """
        Returns the non-UNK special IDs. (Since UNK counts as a vocabulary type, it should not be included anyway.)
        """
        assert is_dataclass(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if issubclass(field.type, Specials):
                yield from iter(value)
            elif field.type == int:
                yield value
            else:
                raise TypeError(field.type)

    def __iter_keys__(self) -> Iterator[tuple[str,str]]:
        """
        Returns field names in order, and their nested form. For example, a Specials structure

            {
                CLS
                text {
                    BOS
                    EOS
                }
                img {
                    IMG_START
                    IMG_END
                }
            }

        returns

            ("CLS",       "CLS")
            ("BOS",       "text.BOS")
            ("EOS",       "text.EOS")
            ("IMG_START", "img.IMG_START")
            ("IMG_END",   "img.IMG_END")

        This list L satisfies

            L == [(path[path.rfind(".")+1:], path) for _,path in L]
        """
        assert is_dataclass(self)
        for field in fields(self):
            if issubclass(field.type, Specials):
                yield from ((name, field.name + "." + path) for name, path in getattr(self, field.name).__iter_keys__())
            elif field.type == int:
                yield field.name, field.name
            else:
                raise TypeError(field.type)

    # The below methods are (1) not public so they don't clutter autocompletion of specials, and (2) not abstract because otherwise you couldn't have partial specials objects.

    def _hfSpecialsMap(self) -> dict[str,str]:  # {"bos_token": "BOS", "eos_token": "EOS", ...}
        from ..interfaces.huggingface import AutoSpecials
        return AutoSpecials.fromTktkt(self, specials_formatter=lambda x: x).special_tokens_map

    def _singleSentenceTemplate(self, ids: list[int]) -> list[int]:
        return ids

    def _pairedSentenceTemplate(self, ids_1: list[int], ids_2: list[int]) -> list[int]:
        return ids_1 + ids_2

    # TODO: We may want to introduce chat templates too. For inspiration, see
    #       https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py


@dataclass
class NoSpecials(Specials):
    pass


########################################################################################################################


UnidentifiedVocab = Iterable[str]  # Vocabulary without identifiers, but in some order.
WithSpecials = TypeVar("WithSpecials", bound=Specials)  # There is a bug where this TypeVar pretends to BE its bound rather than USE its bound. You can still get type completion for .specials, but you should manually annotate the vocab as "v: Vocab[YourSpecialsType]"... https://youtrack.jetbrains.com/issue/PY-49816/Type-inference-fails-when-using-bound-typevar-inference-through-TypeT


@dataclass
class SpecialsExtended(Generic[WithSpecials], metaclass=_ProhibitSubclassing):
    """Purely for bundling Specials with an UNK. This should NOT be used to define a new class of Specials."""
    specials: WithSpecials
    unk: Optional[int] = 0


class Vocab(dict[str, int], Generic[WithSpecials]):

    # Design note: there are two reasons why UNK is a property of the Vocab and not of the Specials.
    #   1. Semantics: an UNK actually refers to the content of the input, unlike actual specials which are
    #      purely there for the extra embedding. UNK is the type for strings that have no other type in the vocabulary.
    #      At the same time, this means UNK is not in the vocabulary because it has no one representation.
    #   2. Implementation: all the ways you could enforce having an UNK field are flawed.
    #       - You could add UNK as a field to the Specials base class, but then that means
    #         every nested set of Specials would also have an UNK and this is disallowed.
    #       - You could have a check in Vocab that hasattr(specials, "UNK") so that the top-level field is
    #         always there, but (1) it is not statically enforced and thus TkTkT can't see the field when
    #         it wants to use specials.UNK, and (2) it encourages people to define their Specials sets
    #         with an UNK which makes those sets unreusable (at least side-by-side).
    #       - You could have a wrapper class like
    #               @dataclass
    #               class SpecialsWithUNK(Specials):
    #                   UNK: bool
    #         but the problem with this is that it encourages inheritance from SpecialsWithUNK
    #         rather than Specials which again encourages people to write unreusable sets of Specials.

    def __init__(self, ordered_types: UnidentifiedVocab, specials: WithSpecials, unk_id: Optional[int]=0):
        """
        A vocabulary with separate subword types and specials.

        :param ordered_types: the types the tokeniser can produce. CANNOT INCLUDE specials or UNK. If you are coming
                              from a framework like HuggingFace's where the vocabulary does contain specials, please
                              see the AutoVocab class for converting it to a Vocab for you.
        :param specials: any object that has (nested) integer fields with unique variable names.
                         If the IDs only take on binary values in {-1,+1}, they will be subdivided into all +1 specials
                         (which go in front of the rest of the vocabulary) and all -1 specials (which go after the rest
                         of the vocabulary), in order of appearance of their fields in the Specials object.
                         Otherwise, the IDs are used as-is.
        :param unk_id: the ID of the unknown type. Must be 0 in the case of relative identifiers.
                       Can be None, e.g. in tokenisers with byte-based preprocessors.
        """
        # Initialise dictionary
        super().__init__()

        # Initialise fields on top of dictionary
        self.UNK                     = unk_id
        self.specials: WithSpecials  = None
        self.inverse: dict[int, str] = None
        self.__constructor_arguments: tuple[WithSpecials, int] = (deepcopy(specials), unk_id)
        self.__next_id         = -1

        # Figure out if the user wanted absolutely or relatively specified specials.
        special_ids = list(specials)
        if areNotAbsoluteSpecials(specials):  # Negatives and/or duplicates only appear in the relative case.
            self._fromRelativeSpecials(ordered_types, specials)  # Has its own assertions, like how UNK == 0.
        else:  # There are no duplicates and no negatives. This is always interpreted as absolute format.
            if unk_id is None:
                warnings.warn("UNK identifier was omitted on purpose. Beware that unless the preprocessor has a universal alphabet, this will cause errors when tokenising unexpected input strings.")
            if len(special_ids) == 1 and unk_id != 0:  # Something that looks like [1, 2] is clearly absolute. Yet, there is a very special case where there is one special and it is +1. If the UNK is 0, it doesn't matter whether relative or absolute is used (it has the same result). If the UNK is not 0, it cannot be relative, but we warn the user that this may not be what they meant.
                warnings.warn("Note: a single special ID of +1 was requested, with UNK != 0. This means that a random type will receive an ID of 0, and the special will have an ID of 1 (absolute interpretation).\\If the +1 meant 'at the start of the vocabulary' (relative interpretation), set it to 0 instead.")
            self._fromAbsoluteSpecials(ordered_types, specials)

    def _fromRelativeSpecials(self, ordered_types: UnidentifiedVocab, specials: WithSpecials):
        assert all(id in {-1,+1} for id in specials), "Cannot have special IDs that are not -1 or +1 when using a relative format (detected whenever there are negative and/or duplicate special IDs)."
        assert self.UNK == 0, "When using relative specials, you must adhere to the TkTkT convention that UNK gets identifier 0, even when your preprocessor has a universal alphabet."

        # Identify all the specials just so you're aware.
        fields_before = []
        fields_after  = []
        for _, recursive_name in specials.__iter_keys__():
            value = getattr_recursive(specials, recursive_name)
            if value == +1:
                fields_before.append(recursive_name)
            elif value == -1:
                fields_after.append(recursive_name)
            else:
                raise ValueError(value)

        # Set lower specials
        specials = deepcopy(specials)  # Any modifications will happen on a private copy the specials.

        next_id = 1  # 0 is reserved for UNK.
        for name in fields_before:
            setattr_recursive(specials, name, next_id)
            next_id += 1

        # Get core vocab
        ordered_types = list(ordered_types)
        next_id += len(ordered_types)

        # Set upper specials
        for name in fields_after:
            setattr_recursive(specials, name, next_id)
            next_id += 1

        self._fromAbsoluteSpecials(ordered_types=ordered_types, absolute_specials=specials)

    def _fromAbsoluteSpecials(self, ordered_types: UnidentifiedVocab, absolute_specials: WithSpecials):
        # Verify that special IDs are not illegal values
        special_ids = list(absolute_specials)
        assert arePositive(special_ids)
        assert areUnique(special_ids)
        assert self.UNK is None or self.UNK >= 0
        assert self.UNK not in special_ids, f"Found UNK ID {self.UNK} in specials: {absolute_specials}"

        # Set all non-special IDs
        next_id = 0
        for typ in ordered_types:
            while next_id == self.UNK or next_id in special_ids:
                next_id += 1

            if typ in self:
                continue
            if typ == "":
                raise EmptyTokenError(f"Tried adding the empty string to the vocabulary (vocabulary was at ID: {next_id}).")

            self[typ] = next_id
            next_id += 1

        # Verify that special IDs after the vocabulary are one span
        upper_special_ids = [special_id for special_id in special_ids if special_id >= next_id]
        if upper_special_ids:
            assert areContiguous(upper_special_ids)
            assert min(upper_special_ids) == next_id, f"Requested special ID too high (got {min(upper_special_ids)}, expected {next_id})."

        # Set core fields
        self.specials = absolute_specials
        self.inverse = {v:k for k,v in self.items()}
        self.__next_id = max(max(self.values(), default=-1), max(self.specials, default=-1)) + 1

    def size(self):
        return len(self) + len(list(self.specials)) + 1

    def __repr__(self):
        return f"{self.__class__.__name__}(types={super().__repr__()}, specials={self.specials.__repr__()}, UNK={self.UNK})"

    def unsafe(self, specials_formatter: Callable[[str], str] = None) -> dict[str, int]:
        """
        Turns the specials into strings and returns a dictionary that contains
        both these strings as well as the vocabulary.

        This is unsafe because it allows incompetent programmers to mistakenly
        match input strings against these formatted specials, or even insert them in
        texts, which allows for injection attacks.
        """
        if specials_formatter is None:  # We choose, and we choose to surround by [ ].
            specials_formatter = lambda s: "[" + s + "]"

        specials_mapping = {
            specials_formatter("UNK"): self.UNK
        } | {
            specials_formatter(name): getattr_recursive(self.specials, recursive_name)
            for name,recursive_name in self.specials.__iter_keys__()
        }
        if intersect_dicts(self, specials_mapping):
            raise ValueError(f"Some (formatted) specials are already part of the vocabulary: {set(intersect_dicts(self, specials_mapping).keys())}")

        union = self | specials_mapping
        return {k: union[k] for k in sorted(union.keys(), key=union.get)}

    # TODO: For the below operations, we need to keep some kind of changelog / transaction journal that allows
    #       easy interpretation by a function that modifies an old embedding matrix into a new one.

    def settle(self):
        """
        Let identifiers sink to the first available free integer.
        If the vocabulary was pruned since its instantiation, this method will make the IDs contiguous again. In this
        sense, you could call this method "compactification".
        If the vocabulary was extended but the original constructor got specials that needed to be at the end, they are
        re-inserted where they were intended.
        """
        types = sorted(self.keys(), key=self.get)
        new_specials, new_unk = self.__constructor_arguments
        new_specials, new_unk = repairAbsoluteSpecials(len(types), new_specials, new_unk)

        # Clears the internal dictionary, and re-initialise all other fields.
        self.clear()
        self.__init__(types, specials=new_specials, unk_id=new_unk)

    def add(self, type: str):
        self[type] = self.__next_id
        self.inverse[self.__next_id] = type
        self.__next_id += 1

    def pop(self, type: str) -> int:
        id = super().pop(type)
        self.inverse.pop(id)
        return id

    def popitem(self) -> tuple[str,int]:
        type, id = super().popitem()
        self.inverse.pop(id)
        return type, id


SubwordCollection = Union[set[str], Vocab]
ExistingSpecialStrings = dict[str, str]  # Mapping like "BOS" -> "<s>" to link Specials to surface strings in the tokeniser's vocab.


@dataclass
class AutoVocabSpecs(Generic[WithSpecials]):
    specials_template: WithSpecials
    special_to_string: ExistingSpecialStrings


class AutoVocab:
    """
    Turns a flat vocabulary with PRE-ASSIGNED identifiers into a Vocab object.

    The name of this class is a misnomer because AutoXYZ refers to classes that generate PretrainedXYZ objects which are
    part of the HuggingFace ecosystem, e.g. AutoModel and AutoTokenizer, and AutoSpecials in TkTkT. This is unfortunate
    but I have no better way name currently.
    """

    @staticmethod
    def fromTokenizer(tokenizer: PreTrainedTokenizerBase, specials_specification: AutoVocabSpecs[WithSpecials]) -> Vocab[WithSpecials]:
        """
        Extracts identifiers from a HuggingFace vocabulary (which has to be attached to a tokeniser because SpecialTokensMixin
        delegates the lookup of its own identifiers to the PreTrainedTokenizerBase.convert_tokens_to_ids method).

        Upside of having a HF tokeniser is that they tell you the specials and their IDs.
        Downside is that a lot of HF tokenisers have more or less unique sets of specials (either because they've set
        some of the canonical variables to None, or because they've set them the same surface form and thus the same ID).

        What this means is that although you can figure out (using certain priority heuristics, e.g. when BOS and EOS are
        the same token, it should count as BOS) which specials a tokeniser has, it is not possible to know at type-checking-time
        (i.e. statically) the subclass of Specials that fits a tokeniser given just the tokeniser, and thus you need to
        declare the specials as a separate argument anyway, rather than just inferring them.
        """
        # Assumptions about the HF framework.
        assert set(tokenizer.all_special_tokens) == set(tokenizer.special_tokens_map.values())
        assert (tokenizer.unk_token is None) == ("unk_token" not in tokenizer.special_tokens_map.keys())

        # Verification step: we can't use the tokeniser's list of specials to statically type the result of this function,
        #                    BUT, even if the user is the one doing the typing, we can still use the tokeniser's specials
        #                    to verify that the user did catch every surface form that the tokeniser identifies as special.
        unsafe_vocab = tokenizer.get_vocab()
        hf_specials = {typ: unsafe_vocab[typ] for typ in tokenizer.all_special_tokens}  # all_special_tokens is a list of surface types ['<s>', '</s>', '<unk>', '<pad>', '<mask>']. It does not contain duplicates, unlike HF's variable-to-surface mapping special_tokens_map {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", ...}.

        # - Before we check that all of HF's surface forms are in the user's dict, we remove HF's UNK surface form, but only if (1) it exists and (2) it is unique. Indeed, some HF tokenisers don't even define it, and others alias it with other specials as a way to not define it.
        unk_surface_form_unique = None
        if tokenizer.unk_token is not None:
            special_to_string_hf = dict(tokenizer.special_tokens_map)
            unk_surface_form = special_to_string_hf.pop("unk_token")
            assert unk_surface_form == tokenizer.unk_token
            if unk_surface_form not in special_to_string_hf.values():  # We only register UNK as a separate token if it doesn't share its surface form.
                unk_surface_form_unique = unk_surface_form
                hf_specials.pop(unk_surface_form_unique)  # Don't consider it part of the specials.

        # - Do the check.
        assert set(hf_specials.keys()).issubset(set(specials_specification.special_to_string.values())), "Every subword indicated by the HF tokeniser as special must be mapped to a field."

        return AutoVocab.fromStrings(
            unsafe_vocab=unsafe_vocab,
            specials_specification=specials_specification,
            unk_type=unk_surface_form_unique
        )

    @staticmethod
    def fromStrings(unsafe_vocab: dict[str,int], specials_specification: AutoVocabSpecs[WithSpecials], unk_type: Optional[str]) -> Vocab[WithSpecials]:
        assert "UNK" not in specials_specification.special_to_string.keys() and unk_type not in specials_specification.special_to_string.values(), "UNK does not count as a special. It is specified as a separate argument."
        assert set(map(fst,specials_specification.specials_template.__iter_keys__())) == set(specials_specification.special_to_string.keys()), "The keys of the 'special_to_string' mapping should match the fields of the Specials template."  # Could technically be a subset of the mapping, but I'd like the connection between the two arguments to be very strict.
        assert set(specials_specification.special_to_string.values()).issubset(set(unsafe_vocab.keys())), "The values of the 'special_to_string' mapping should all be types in the given set of types."
        assert areUnique(unsafe_vocab.values())
        assert arePositive(unsafe_vocab.values())
        assert areContiguous(sorted(unsafe_vocab.values()))

        # Remove IDs from vocab and assign them to the Specials template to make it absolute. (Modifies copies of both objects.)
        unsafe_vocab      = dict(unsafe_vocab)
        specials_template = deepcopy(specials_specification.specials_template)
        for name,recursive_name in specials_template.__iter_keys__():
            setattr_recursive(specials_template, recursive_name, unsafe_vocab.pop(specials_specification.special_to_string[name]))
        unk_id = unsafe_vocab.pop(unk_type) if unk_type else None

        # Finally, build the vocabulary.
        vocab = Vocab(
            ordered_types=sorted(unsafe_vocab.keys(), key=unsafe_vocab.get),
            specials=specials_template,
            unk_id=unk_id
        )

        # As long as there were no gaps in the given vocabulary, reassigning IDs to the sequence of types should result in the exact same identifiers.
        for typ in vocab:
            assert vocab[typ] == unsafe_vocab[typ], f"For some reason, type {typ} got assigned a different ID in the new vocabulary: {vocab[typ]} (was {unsafe_vocab[typ]})."

        return vocab


def areNotAbsoluteSpecials(specials: Specials) -> bool:
    """
    Absolute specials have only positive IDs and have no duplicates.
    Now, whether the specials are relative (only +1/-1), that needs further checking.
    """
    special_ids = list(specials)
    return any(id < 0 for id in special_ids) or not areUnique(special_ids)


def repairAbsoluteSpecials(n_types: int, specials: WithSpecials, unk_id: Optional[int]) -> tuple[WithSpecials, Optional[int]]:
    """
    In cases where you have absolute specials but the subword vocabulary is too small for the given IDs, you need to
    decrease the upper IDs.
    """
    specials = deepcopy(specials)
    pinned_ids = ({"": unk_id} if unk_id is not None else dict()) | {nested_field: getattr_recursive(specials, nested_field) for _, nested_field in specials.__iter_keys__()}

    for field, id in sorted(pinned_ids.items(), key=snd):
        if id > n_types:  # => downshift
            if field:
                setattr_recursive(specials, field, n_types)
            else:
                unk_id = n_types
        n_types += 1

    return specials, unk_id


def findSpecialRanges(specials: Specials) -> list[tuple[int,int]]:
    """
    It is sometimes useful in language models to be able to check for specials using a range check rather than looping
    through equality checks. Usually you have a top range of specials and a bottom range of specials, although specials
    can in principle exist anywhere in the vocab.

    Returns the lower and (inclusive) upper ID of these ranges.
    UNK does not count as a special.
    """
    assert not areNotAbsoluteSpecials(specials)
    special_ids = sorted(list(specials))

    ranges = []
    start = 0
    while start < len(special_ids):
        offset = 0
        while not areContiguous(special_ids[start:len(special_ids)-offset]):
            offset += 1

        ranges.append((special_ids[start], special_ids[len(special_ids)-offset-1]))
        start = len(special_ids) - offset

    return ranges
