"""
"DeL" stands for "Derivational Leverager". It was used to fine-tune BERT into DelBERT by Valentin Hofmann in
his superbizarre paper: https://aclanthology.org/2021.acl-long.279.pdf.

Reimplementation of the source code found at
    https://github.com/valentinhofmann/superbizarre
"""
from pathlib import Path
from typing import Union, Optional, Iterable
from abc import ABC, abstractmethod

from ...interfaces.tokenisers import *
from ...util.iterables import streamLines

DEL_DATA = Path(__file__).resolve().parent / "data"


def is_vowel(char):
    return char.lower() in 'aeiou'

def is_cons(char):
    return char.lower() in 'bdgptkmnlrszfv'


class Derivator(ABC):

    def __init__(self, prefices: Iterable[str], suffices: Iterable[str], stems: Iterable[str]):
        self.prefices = list(prefices)
        self.suffices = list(suffices)
        self.stems    = set(stems)

    @classmethod  # Output doesn't differ between instances of the same class, but superclass needs to be able to call "whatever implementation the class overrides this method with" which you can't do with a static method (since ParentClass.method() won't call the overriding implementation).
    @abstractmethod
    def _pathToDefaultPreficesAndSuffices(cls) -> tuple[Path,Path]:
        pass

    @classmethod
    def fromFiles(cls, path_prefices: Path=None, path_suffices: Path=None, path_stems: Path=None):
        """
        Can be overridden to support other constructors, but this default will work for most languages.
        The reason for not having this method itself as the constructor is because it's bad design to expect subclasses
        to have the same interface for their __init__ method as the superclass. That's not what __init__ is for.

        If you want more arguments, you can put them in __init__'s signature and override this fromFiles method to give
        appropriate values to those arguments, changing the interface for __init__ whilst keeping that of fromFiles stable.
        Hence, users of this class can always expect to give the same three files.
        """
        # Impute paths
        default_prefices, default_suffices = cls._pathToDefaultPreficesAndSuffices()
        path_prefices = path_prefices or default_prefices
        path_suffices = path_suffices or default_suffices

        # Read files
        prefices = list(streamLines(path_prefices, include_empty_lines=False))
        suffices = list(streamLines(path_suffices, include_empty_lines=False))
        stems    = list(streamLines(path_stems, include_empty_lines=False)) if path_stems is not None else []
        return cls(prefices, suffices, stems)

    @abstractmethod
    def invertDerivation(self, word: str) -> tuple[list[str], str, list[str]]:
        """
        Go from derivation back to prefices, root and suffices.
        For example, go from the derivation "animation" back to ([], "animate", ["ion"]).
        """
        pass


class EnglishDerivator(Derivator):
    """
    Core implementation. Is to DeL what BTE is to BPE-knockout.
    """

    @classmethod
    def _pathToDefaultPreficesAndSuffices(cls) -> tuple[Path,Path]:
        return (DEL_DATA / "english_prefices.txt", DEL_DATA / "english_suffices.txt")

    def segment(self, word: str) -> Optional[tuple[list[str], str, list[str]]]:
        """
        Core method.
        Produces prefices, root and suffices when applicable.
        """
        found_prefices = []
        did_find_prefix = True

        # Outer loop to check prefixes
        while did_find_prefix:
            found_suffices = []
            did_find_suffix = True
            root = word

            # Inner loop to check suffixes
            while did_find_suffix:
                if not root:
                    break

                # Termination conditions:
                if root in self.stems:
                    return found_prefices, root, found_suffices[::-1]
                elif len(root) >= 3 and is_cons(root[-1]) and found_suffices:
                    if is_vowel(found_suffices[-1][0]) and root + "e" in self.stems:
                        root = root + "e"
                        return found_prefices, root, found_suffices[::-1]
                    elif root[-1] == root[-2] and is_vowel(root[-3]) and root[:-1] in self.stems:
                        root = root[:-1]
                        return found_prefices, root, found_suffices[::-1]

                # Need to find suffix to stay in inner loop
                did_find_suffix = False
                found_sfx = ""
                for sfx in self.suffices:
                    if root.endswith(sfx):
                        did_find_suffix = True
                        if sfx == "ise":
                            found_sfx = "ize"
                        else:
                            found_sfx = sfx
                        found_suffices.append(found_sfx)
                        root = root[:-len(sfx)]
                        break

                # Check for special phonological alternations
                if root[-4:] == "ific" and found_sfx in {"ation", "ate"}:
                    root = root[:-4] + "ify"
                elif root[-1:] == "i" and found_sfx == "ness":
                    root = root[:-1] + "y"
                elif root[-4:] == "abil":
                    root = root[:-4] + "able"

            # Need to find prefix to stay in outer loop
            did_find_prefix = False
            for pfx in self.prefices:
                if word.startswith(pfx):
                    # Check addition of false prefixes
                    did_find_prefix = True
                    if pfx in {"im", "il", "ir"}:
                        found_prefix = "in"
                    else:
                        found_prefix = pfx
                    found_prefices.append(found_prefix)
                    word = word[len(pfx):]
                    break

        return None

    def invertDerivation(self, word: str) -> tuple[list[str], str, list[str]]:
        """
        Wrapper around .segment() that handles the None case.

        This method used to have a `mode` parameter that could be one of {"morphemes", "bundles", "roots"}. Now this
        method is just the "morphemes" version because it is the only one called in Valentin's repo.
        """
        try:
            prefices, root, suffices = self.segment(word)  # This assignment is what may trigger a ValueError (not enough values to unpack)
            return prefices, root, suffices
        except ValueError:
            return [], word, []

    def invertDerivation_bundled(self, word: str) -> tuple[str, str, str]:
        try:
            prefices, root, suffices = self.segment(word)
            return "".join(p + "_" for p in prefices), root, "".join("_" + s for s in suffices)
        except ValueError:
            return "", word, ""

    def invertDerivation_root(self, word: str) -> str:
        try:
            _, root, _ = self.segment(word)
            return root
        except ValueError:
            return word

    def derive(self, word: str, mode: str) -> Union[tuple[list[str], str, list[str]], tuple[str, str, str], str]:
        if mode == "morphemes":
            return self.invertDerivation(word)
        elif mode == "bundles":
            return self.invertDerivation_bundled(word)
        elif mode == "roots":
            return self.invertDerivation_root(word)
        else:
            raise ValueError("Derivative mode unrecognised:", mode)

    # def tokenize(self, word_list: Union[str, list[str]], mode="bundles"):
    #     """
    #     Serialises the output of .derive() for many words into one big list.
    #     This isn't actually used anywhere.
    #     """
    #     if isinstance(word_list, str):  # It's just a sentence.
    #         word_list = word_list.split()
    #
    #     output = []
    #     for word in word_list:
    #         if mode == "roots":
    #             output.append(self.derive_root(word))
    #         if mode == "bundles":
    #             output.extend([s for s in self.derive_bundled(word) if s])
    #         if mode == "morphemes":
    #             prefices, root, suffices = self.derive(word)
    #             output.extend(prefices)
    #             output.append(root)
    #             output.extend(suffices)  # Doesn't take into account e.g. BERT's "##".
    #
    #     return output


class DeL(Tokeniser):
    """
    TkTkT wrapper around a derivator.
    """

    def __init__(self, preprocessor: Preprocessor, derivator: Derivator, prefix_separator: str="-"):
        super().__init__(preprocessor)
        self.derivator = derivator
        self.prefix_sep = prefix_separator

    def tokenise(self, pretoken: str) -> Tokens:
        # TODO: Split off any SoW/EoW from the pretoken, because otherwise this doesn't work.
        prefices, root, suffices = self.derivator.invertDerivation(pretoken)

        tokens = []
        for p in prefices:
            tokens.append(p)
            tokens.append(self.prefix_sep)
        tokens.append(root)
        for s in suffices:
            tokens.append("##" + s)  # TODO: Uses BERT's convention of ##. Should allow any convention.

        return tokens
