from pathlib import Path

from collections import defaultdict
import json
from typing import Iterable

from pickybpe.utils import Token, PairCounts, PathLike
from pickybpe.vocabularisation import BPETrainer as _BPETrainerBase

from ...interfaces import Preprocessor
from .decomposing import ScaffoldBPE
from .vocabularisation import _VocabulariserWithChizhovBackend

__all__ = ["ScaffoldBPE", "ScaffoldBPEVocabulariser"]

from ...interfaces.vocabulariser import UnidentifiedVocab


class _ChizhovBackend_ScaffoldBPE(_BPETrainerBase):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
        super().__init__(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            ensured_vocabulary=preprocessor.getAlphabet(),
            max_type_length=max_type_length,
            include_specials=False
        )
        self._scaffolds_and_causes: dict[str,list[str]] = defaultdict(list)
        self._marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self._marker.atomise(word)

    def _scrutinize_parent_after_merge(self, parent: Token, child: Token, pair_frequency: int, pairs: PairCounts):
        _, next_pair_frequency = pairs.get_argmax()
        if parent.freq < next_pair_frequency:
            self._scaffolds_and_causes[parent.str].append(child.str)
            self.actual_vocab_size -= 1

    def _dump(self, path: PathLike):
        folder = Path(path).resolve()
        if folder.suffix:
            folder = folder.parent

        # Dump extended vocab and merges
        super()._dump(folder)

        # Dump scaffold types with diagnostics
        with open(folder / "ablations.json", "w", encoding="utf-8") as handle:
            json.dump({
                scaffold_parent: {
                    "id": self.str2token[scaffold_parent].id,
                    "accusers": {
                        child_type: self.str2token[child_type].id
                        for child_type in children
                    }
                }
                for scaffold_parent, children in self._scaffolds_and_causes.items()
            }, handle, indent=4)


class ScaffoldBPEVocabulariser(_VocabulariserWithChizhovBackend):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: int, max_type_length: int):
        super().__init__(name="scaffoldbpe", preprocessor=preprocessor, backend=_ChizhovBackend_ScaffoldBPE(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            max_type_length=max_type_length,
            character_coverage=character_coverage
        ))

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:  # Loads only the ablated types.
        path = Path(file_or_folder).resolve()
        if path.is_dir():
            path = path / "ablations.json"

        with open(path, "r", encoding="utf-8") as handle:
            return [t for t in json.load(handle).keys()]

    # @classmethod
    # def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
    #     if file_or_folder.is_file():
    #         file_or_folder = file_or_folder.parent
    #
    #     with open(file_or_folder / "vocab.json", "r", encoding="utf-8") as handle:
    #         vocab = json.load(handle)
    #         all_types = set(vocab.keys())
    #
    #     with open(file_or_folder / "ablations.json", "r", encoding="utf-8") as handle:
    #         scaffold_types = set(json.load(handle).keys())
    #
    #     return sorted(all_types - scaffold_types, key=vocab.get)
