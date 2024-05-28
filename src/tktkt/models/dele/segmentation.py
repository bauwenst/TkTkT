"""
"DeL" stands for "Derivational Leverager". It was used to fine-tune BERT into DelBERT by Valentin Hofmann in
his superbizarre paper: https://aclanthology.org/2021.acl-long.279.pdf.

Reimplementation of the source code found at
    https://github.com/valentinhofmann/superbizarre

FIXME: Verify whether the GitHub list is an intersection with BERT's vocab.
       The way the list is constructed is by starting from a gigantic corpus and running DeL with BERT's entire vocab
       as stem set. Afterwards, we check which subwords were actually used as stems, and reduce the stem set to only
       that subset. By using this reduced set, we can now expand the input domain to any string, and we are sure that
       we won't accidentally recognise prefices/suffices since we now know which stems can be trusted as real.
       |
       So basically, DeL should be able to run with and without a "safe stem" list.
"""
from pathlib import Path
from typing import Union, List, Tuple, Optional

from ...interfaces.preparation import Preprocessor
from ...interfaces.tokeniser import Tokeniser

THIS_FOLDER = Path(__file__).resolve().parent
DEFAULT_PREFICES = THIS_FOLDER / "english_prefices.txt"
DEFAULT_SUFFICES = THIS_FOLDER / "english_suffices.txt"


def is_vowel(char):
    return char.lower() in 'aeiou'

def is_cons(char):
    return char.lower() in 'bdgptkmnlrszfv'


class EnglishDerivator:
    """
    Core implementation. Is to DeL what BTE is to BPE-knockout.
    """

    def __init__(self, path_prefices: Path=DEFAULT_PREFICES, path_suffices: Path=DEFAULT_SUFFICES, path_stems="stems.txt"):
        self.stems = set()

        with open(path_prefices, "r", encoding="utf-8") as handle:
            self.prefices = [line.strip().lower() for line in handle if line.strip()]

        with open(path_suffices, "r", encoding="utf-8") as handle:
            self.suffices = [line.strip().lower() for line in handle if line.strip()]

        with open(path_stems, "r", encoding="utf-8") as handle:
            self.stems = {line.strip().lower() for line in handle if line.strip()}

    def segment(self, word: str) -> Optional[List[str], str, List[str]]:
        """
        Core method.
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

    def derive(self, word: str) -> Tuple[List[str], str, List[str]]:
        """
        Exception-safe wrapper around .segment().

        Used to have a `mode` parameter that could be one of {"morphemes", "bundles", "roots"}. Now this method is just
        the "morphemes" version because it is the only one called in Valentin's repo.
        """
        try:
            prefices, root, suffices = self.segment(word)  # This assignment is what may trigger a ValueError (not enough values to unpack)
            return prefices, root, suffices
        except ValueError:
            return [], word, []

    def derive_bundled(self, word: str) -> Tuple[str, str, str]:
        try:
            prefices, root, suffices = self.segment(word)
            return "".join(p + "_" for p in prefices), root, "".join("_" + s for s in suffices)
        except ValueError:
            return "", word, ""

    def derive_root(self, word: str) -> str:
        try:
            _, root, _ = self.segment(word)
            return root
        except ValueError:
            return word

    def tokenize(self, word_list: Union[str, List[str]], mode="bundles"):
        """
        Serialises the output of .derive() for many words into one big list.
        This isn't actually used anywhere.
        """
        if isinstance(word_list, str):  # It's just a sentence.
            word_list = word_list.split()

        output = []
        for word in word_list:
            if mode == "roots":
                output.append(self.derive_root(word))
            if mode == "bundles":
                output.extend([s for s in self.derive_bundled(word) if s])
            if mode == "morphemes":
                prefices, root, suffices = self.derive(word)
                output.extend(prefices)
                output.append(root)
                output.extend(suffices)

        return output


class DeL(Tokeniser):
    """
    TkTkT wrapper around a derivator.
    """

    def __init__(self, preprocessor: Preprocessor, prefix_separator: str="-"):
        super().__init__(preprocessor)
        self.derivator = EnglishDerivator()
        self.prefix_sep = prefix_separator

    def tokenise(self, pretoken: str) -> List[str]:  # TODO: Uses BERT's convention of ##. Should allow any convention.
        # TODO: Split off any SoW/EoW pretoken, because otherwise this doesn't work.
        prefices, root, suffices = self.derivator.derive(pretoken)

        tokens = []
        for p in prefices:
            tokens.append(p)
            tokens.append(self.prefix_sep)
        tokens.append(root)
        for s in suffices:
            tokens.append("##" + s)

        return tokens
