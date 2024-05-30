"""
TODO: Find Valentin's hand-crafted list of prefixes and suffixes.
FIXME: Definitely needs special handling for handling SoW/EoW in the base tokeniser.
"""
from typing import Iterable, List, Type
from pathlib import Path

from .segmentation import EnglishDerivator, Derivator
from ...interfaces.tokeniser import TokeniserWithFiniteIdRange, TokeniserWithFiniteTypeDomain
from ...util.lists import fileToList
from ...util.timing import datetimeDashed
from ...files.paths import DataPaths

FILESTEM_PREFICES = "prefices"
FILESTEM_SUFFICES = "suffices"
FILESTEM_STEMS    = "stems"


class DelTrainer:
    """
    DeL is a tokeniser that requires a list of prefices, list of suffices and list of stems, all of which have to
    belong to the vocabulary of an existing tokeniser.

    You start with a list of prefices and suffices from the language. There are two ways of getting the list of valid stems:
        - Coarse-grained: all types in the vocabulary larger than 4 characters are valid stems.
        - Fine-grained: all types in the vocabulary larger than 4 characters THAT result from applying the DeL process
                        to any of the words in a given corpus are valid stems.
    """

    def __init__(self, derivator_class: Type[Derivator], base_tokeniser: TokeniserWithFiniteTypeDomain):
        self.model_class = derivator_class
        self.tokeniser = base_tokeniser

    def load(self, folder: Path):
        return self.model_class.fromFiles(
            folder / (FILESTEM_PREFICES + ".txt"),
            folder / (FILESTEM_SUFFICES + ".txt"),
            folder / (FILESTEM_STEMS + ".txt")
        )

    def save(self, prefices: Iterable[str], suffices: Iterable[str], stems: Iterable[str], stem_suffix: str=""):
        model_folder = DataPaths.pathToModels() / "del" / (
                "del_"
                + f"{self.model_class.__name__}+{self.tokeniser.__class__.__name__}_"
                + (f"{stem_suffix}_" if stem_suffix else "")
                + datetimeDashed()
        )
        model_folder.mkdir(exist_ok=True, parents=True)

        with open(model_folder / (FILESTEM_PREFICES + ".txt"), "w", encoding="utf-8") as handle:
            for prefix in sorted(prefices):
                handle.write(prefix + "\n")

        with open(model_folder / (FILESTEM_SUFFICES + ".txt"), "w", encoding="utf-8") as handle:
            for suffix in sorted(suffices):
                handle.write(suffix + "\n")

        with open(model_folder / (FILESTEM_STEMS + ".txt"), "w", encoding="utf-8") as handle:
            for stem in sorted(stems):
                handle.write(stem + "\n")

        return model_folder

    def train_coarse(self, length_limit: int) -> Path:
        untrained_derivator = self.model_class.fromFiles()
        return self.save(
            (p for p in untrained_derivator.prefices if self.tokeniser.hasType(p)),  # FIXME: In a BPE vocab you might see these with start-of-word, and since DeL has a start-of-word for each prefix in a sequence, you want to explicitly check that the SoW version exists!
            (s for s in untrained_derivator.suffices if self.tokeniser.hasType(s)),  # FIXME: This is more difficult, since only one suffix is allowed to have an EoW and if the suffix doesn't exist without it, it can't be used in that case.
            (t for t in self.tokeniser.types() if len(t) >= length_limit),
            stem_suffix="coarse"
        )

    def train_fine(self, sentences: Iterable[str]) -> Path:
        coarse_model = self.train_coarse(length_limit=4)
        coarse_model = self.load(coarse_model)

        used_stems = set()
        for sentence in sentences:
            for word in self.tokeniser.preprocessor.do(sentence):  # FIXME: Watch out for markers.
                pre, root, suf = coarse_model.invertDerivation(word)
                if pre or suf:
                    used_stems.add(root)

        return self.save(
            coarse_model.prefices,
            coarse_model.suffices,
            used_stems,
            stem_suffix="fine"
        )
