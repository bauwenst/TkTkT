from typing import Iterable
from pathlib import Path
from collections import defaultdict
import json

from pickybpe.utils import Token, PairCounts, PathLike
from pickybpe.vocabularisation import BPETrainer as _BPETrainerBase, EventType

from ...interfaces import Preprocessor
from ...interfaces.vocabularisers import UnidentifiedVocab
from .decomposing import ScaffoldBPE, CacheableAblatedBPEArtifacts
from .vocabularisation import _VocabulariserWithChizhovBackend, CacheableBPEArtifacts, _ChizhovTrainingContext

__all__ = ["ScaffoldBPE", "ScaffoldBPEVocabulariser"]


class _ChizhovBackend_ScaffoldBPE(_BPETrainerBase):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
        super().__init__(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            ensured_vocabulary=preprocessor.getAlphabet(),
            max_type_length=max_type_length,
            include_specials=False
        )
        self._marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self._marker.atomise(word)

    def _initialize_state(self) -> _ChizhovTrainingContext:
        self._scaffolds_and_causes: dict[str,list[str]] = defaultdict(list)
        return super()._initialize_state()
        
    def _scrutinize_parent_after_merge(self, parent: Token, child: Token, pair_frequency: int, state: _ChizhovTrainingContext):
        _, next_pair_frequency = pairs.get_argmax()
        if parent.freq < next_pair_frequency:
            self._scaffolds_and_causes[parent.str].append(child.str)
            state.actual_vocab_size -= 1

    def _dump(self, state: _ChizhovTrainingContext, path: PathLike):
        folder = Path(path).resolve()
        if not folder.is_dir():
            folder = folder.parent

        # Dump extended vocab and merges
        # super()._dump(folder)
        # - Vocab
        CacheableBPEArtifacts._storeTypes(folder, [
            typ.str for typ in sorted(filter(lambda token: token != self.unk_token, state.str2token.values()), key=lambda token: token.id)
        ])

        # Merges
        def validate_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")
            return token

        CacheableBPEArtifacts._storeMerges(folder,
            (
                [validate_characters(part.str) for part in parts]
                for event_type, parts, _ in state.events if event_type == EventType.MERGE
            )
        )

        # Dump scaffold types with diagnostics
        with open(folder / "ablations.json", "w", encoding="utf-8") as handle:
            json.dump({
                scaffold_parent: {
                    "id": state.str2token[scaffold_parent].id,
                    "accusers": {
                        child_type: state.str2token[child_type].id
                        for child_type in children
                    }
                }
                for scaffold_parent, children in self._scaffolds_and_causes.items()
            }, handle, indent=4)


class ScaffoldBPEVocabulariser(_VocabulariserWithChizhovBackend[CacheableAblatedBPEArtifacts]):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: int, max_type_length: int):
        super().__init__(preprocessor=preprocessor, backend=_ChizhovBackend_ScaffoldBPE(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            max_type_length=max_type_length,
            character_coverage=character_coverage
        ))

    def _cacheSubfolder(self) -> str:
        return "scaffoldbpe"

    def _cacheType(self):
        return CacheableAblatedBPEArtifacts

    def _dumpToArtifacts(self, dump_path: Path) -> CacheableAblatedBPEArtifacts:
        with open(dump_path / "ablations.json", "r", encoding="utf-8") as handle:
            deleted = set(json.load(handle).keys())

        return CacheableAblatedBPEArtifacts(
            types=CacheableBPEArtifacts._loadTypes(dump_path),
            merges=CacheableBPEArtifacts._loadMerges(dump_path),
            ablated_types=deleted
        )

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
