from typing import Iterator
from dataclasses import dataclass, is_dataclass, fields

from transformers import PreTrainedTokenizerBase


class ProhibitDeclaringConstructor(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        if namespace.get("__init__",
                         None) is not None:  # namespace is basically what is written in code, but as a dictionary.
            raise AssertionError(
                "Specials should have no __init__ constructor. They should be constructed through the @dataclass decorator.")


class Specials(metaclass=ProhibitDeclaringConstructor):  # This metaclass prevents the user from writing 'def __init__(self)' which means they are more or less forced to use a dataclass.

    def __post_init__(self):  # This method is only called by @dataclass objects, namely at the end of their implicitly declared constructor. It is inherited, even if the base class is not a dataclass.
        assert is_dataclass(self)
        for field in fields(self):
            if field.type == int:
                value = getattr(self, field.name)
                if isinstance(value, bool) or not isinstance(value, int):  # You can't just ask "not isinstance(value, int)" because bools are ints lmao
                    raise TypeError(f"Special {field.name} was assigned a value of non-int type {type(value)}.")
                if value not in {-1, +1}:
                    raise ValueError(
                        f"Special {field.name} should be initialised with a value of -1 or +1 (got {value}).")
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

    def __iter__(self) -> Iterator[int]:  # Returns the non-UNK special IDs. (Since UNK counts as a vocabulary type, it should not be included anyway.)
        assert is_dataclass(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if issubclass(field.type, Specials):
                yield from iter(value)
            elif field.type == int:
                yield value
            else:
                raise TypeError(field.type)

    def __iter_keys__(self) -> Iterator[str]:
        assert is_dataclass(self)
        for field in fields(self):
            if issubclass(field.type, Specials):
                yield from (field.name + "." + name for name in getattr(self, field.name).__iter_keys__())
            elif field.type == int:
                yield field.name
            else:
                raise TypeError(field.type)

    # No other methods because we want the user to just see all the available specials when they type "."


@dataclass
class NoSpecials(Specials):
    pass


@dataclass
class GenericHuggingfaceSpecials(Specials):
    BOS: int
    EOS: int
    CLS: int
    SEP: int
    PAD: int
    MASK: int


def getSpecialsFromHf(tokenizer: PreTrainedTokenizerBase) -> GenericHuggingfaceSpecials:
    """
    Extracts identifiers from a HuggingFace vocabulary (which has to be attached to a tokeniser because
    SpecialTokensMixin delegates the lookup of its own identifiers to the PreTrainedTokenizerBase.convert_tokens_to_ids method).
    """
    return GenericHuggingfaceSpecials(
        BOS=tokenizer.bos_token_id,
        EOS=tokenizer.eos_token_id,
        CLS=tokenizer.cls_token_id,
        SEP=tokenizer.sep_token_id,
        PAD=tokenizer.pad_token_id,
        MASK=tokenizer.mask_token_id
    )
