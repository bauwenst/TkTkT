from typing import Union, Iterable, Tuple
from pathlib import Path
from dataclasses import dataclass

import json
from collections import defaultdict, Counter

from pickybpe.utils import PathLike
from pickybpe.vocabularisation import logger, EventType, PickyBPETrainer as _PickyBPETrainerBase, BPETrainer as _BPETrainerBase
from pickybpe.segmentation import Token, MergeEventIds, SplitEventIds, SplitResults, PickyBPESegmenter as _PickyBPETokeniserBackend

from ...interfaces import Deserialiser, Vocabulariser, Preprocessor
from ...interfaces.vocabulariser import Vocab, UnidentifiedVocab
from ...interfaces.tokeniser import TokeniserWithVocabDict, Tokens
from ...util.types import NamedIterable


class _ChizhovBackend_BPE(_BPETrainerBase):
    """
    Overrides the way the pickybpe package initially splits words, and how it dumps the vocab/merges.

    The former should actually apply to all TkTkT derivatives that have _BPETrainerBase as a parent class, but unfortunately
    TkTkT is not a dependency of the pickybpe package, and thus we add this functionality to all three derived classes
    (this one, PickyBPE backend, ScaffoldBPE) by repeating the same method implementation every time.
    """

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
        super().__init__(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            ensured_vocabulary=preprocessor.getAlphabet().getCharacters() if preprocessor.getAlphabet() else [],
            max_type_length=max_type_length,
            include_specials=False
        )
        self._marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self._marker.intoCharacters(word)

    def _dump(self, path: PathLike):
        folder = Path(path).resolve()
        if folder.suffix:
            folder = folder.parent
        logger.info(f'Dumping model to {folder.as_posix()}...')

        # Vocab
        from .vocabularisation import BPEVocabulariser
        BPEVocabulariser._storeVocab({
            typ.str: i
            for i, typ in enumerate(sorted(self.str2token.values(), key=lambda token: token.id))
        }, folder)

        # Merges
        def validate_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")
            return token

        BPEVocabulariser._storeMerges(
            (
                [validate_characters(part.str) for part in parts]
                for event_type, parts, _ in self.events if event_type == EventType.MERGE
            ),
            folder
        )
        return folder


class _VocabulariserWithChizhovBackend(Vocabulariser):
    """
    Puts a Vocabulariser interface around the pickybpe package.
    Supports word frequency files, HuggingFace datasets, ... which were not supported by the original codebase.

    This class itself cannot be instantiated because it doesn't know which _dump() implementation the backend uses.
    """

    def __init__(self, name: str, preprocessor: Preprocessor, backend: _BPETrainerBase):
        super().__init__(name=name, preprocessor=preprocessor)
        self._backend = backend

    def _vocabulariseFromPretokenCounts(self, pretoken_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        """Does no preprocessing."""
        folder = self._makeOutputFolder(pretoken_iterable.name)
        return self._backend._fit_from_counts(Counter(dict(pretoken_iterable)), folder, logging_step=100)

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        return self._vocabulariseFromPretokenCounts(self._preprocessWordsToPretokenCounts(word_iterable))

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        return self._vocabulariseFromPretokenCounts(self._preprocessSentencesToPretokenCounts(sentence_iterable))


class BPEVocabulariser_Chizhov(_VocabulariserWithChizhovBackend):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
        super().__init__(name="bpe", preprocessor=preprocessor, backend=_ChizhovBackend_BPE(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            max_type_length=max_type_length
        ))

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        from .vocabularisation import BPEVocabulariser
        return BPEVocabulariser._load(file_or_folder)


class _ChizhovBackend_PickyBPE_SmallFormat(_PickyBPETrainerBase):
    """
    PickyBPE trainer that stores the results of training in two much smaller files than the original codebase.
    It already removes support for specials.

    Handling of other input formats, which does not deal with the actual mechanism of PickyBPE, is done by the
    Vocabulariser class that extends this one.
    """

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, picky_threshold: float, character_coverage: float, max_type_length: int):
        super().__init__(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            ensured_vocabulary=preprocessor.getAlphabet().getCharacters() if preprocessor.getAlphabet() else [],
            max_type_length=max_type_length,
            picky_threshold=picky_threshold,
            include_specials=False
        )
        self._marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self._marker.intoCharacters(word)

    def _dump(self, file: Union[Path, str]):
        folder = Path(file).resolve()
        if folder.suffix:
            folder = folder.parent
        logger.info(f'Dumping model to {folder.as_posix()}...')

        # Vocabulary stores verbose types
        with open(folder / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({
                typ.id: typ.to_dict()
                for typ in sorted(self.str2token.values(), key=lambda token: token.id)
            }, f, indent=4)

        # Event list is a generalisation of the merge list. "+" means token merge, "-" means token split.
        def validate_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")
            return token

        with open(folder / "events.txt", "w", encoding="utf-8") as f:
            for event_type, start, end in self.events:
                if event_type == EventType.MERGE:
                    f.write(f"+ {' '.join(validate_characters(token.str) for token in start)}\n")
                elif event_type == EventType.SPLIT:
                    f.write(f"- {' '.join(validate_characters(token.str) for token in end)}\n")
                else:
                    raise NotImplementedError

        return folder


class PickyBPEVocabulariser(_VocabulariserWithChizhovBackend):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, picky_threshold: float, character_coverage: float, max_type_length: int):
        super().__init__(name="pickybpe", preprocessor=preprocessor, backend=_ChizhovBackend_PickyBPE_SmallFormat(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            max_type_length=max_type_length,
            picky_threshold=picky_threshold,
            character_coverage=character_coverage
        ))

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        vocab = PickyBPEDeserialiser_SmallFormat._simpleVocabularyFromFile(file_or_folder)
        return sorted(vocab.keys(), key=vocab.get)


########################################################################################################################


@dataclass
class Event:
    left: str
    right: str
    is_merge: bool  # If false, it is a split into these two tokens.


class PickyBPEDeserialiser_SmallFormat(Deserialiser):

    @classmethod
    def _simpleVocabularyFromFile(cls, path: Path) -> Vocab:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "vocab.json"

        with open(path, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)
            return {
                token_dict["str"]: int(token_dict["id"])
                for token_dict in vocab.values()
            }

    @classmethod
    def _verboseVocabularyFromFile(cls, path: Path) -> dict[str, Token]:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "vocab.json"

        # Get vocabulary
        with open(path, "r", encoding="utf-8") as f:
            serialised_vocab = json.load(f)

        token_to_object = dict()
        id_to_object    = dict()
        for token_dict in sorted(serialised_vocab.values(), key=lambda item: item[0]):
            token = Token(
                id=token_dict['id'],
                str=token_dict['str'],
                freq=token_dict['freq'],
                special=token_dict['special'],
                present=token_dict['present'],
                # If you'd merge it:
                left=id_to_object[token_dict['left']]   if token_dict['left']  is not None else None,
                right=id_to_object[token_dict['right']] if token_dict['right'] is not None else None,
                # If you'd split it:
                split=[id_to_object[id] for id in token_dict['split']] if len(token_dict['split']) > 1 else None
            )
            token_to_object[token.str] = token
            id_to_object[token.id]     = token

        return token_to_object

    @classmethod
    def _eventsFromFile(cls, path: Path) -> list[Event]:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "events.txt"

        # Get events
        events = []
        with open(path, "r", encoding="utf-8") as handle:
            for event_id, event in enumerate(handle):
                event = event.strip()
                typ, left, right, = event.split(" ")
                if typ == "+":
                    events.append(Event(left, right, True))
                elif typ == "-":
                    events.append(Event(left, right, False))
                else:
                    raise ValueError(f"Encountered line with unknown prefix '{typ}'.")

        return events

    @classmethod
    def _eventsToDataStructures(cls, verbose_vocabulary: dict[str, Token], events: list[Event]) -> tuple[MergeEventIds, SplitEventIds, SplitResults]:
        merge_map = defaultdict(list)
        split_map = defaultdict(list)
        splits = dict()

        for event_id, event in enumerate(events):
            if event.is_merge:
                merge_map[(verbose_vocabulary[event.left], verbose_vocabulary[event.right])].append(event_id)
            else:
                split_map[verbose_vocabulary[event.left + event.right]].append(event_id)
                splits[event_id] = [verbose_vocabulary[event.left], verbose_vocabulary[event.right]]

        return merge_map, split_map, splits


class PickyBPE(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, expanded_vocab: Vocab, events: list[Event]):
        super().__init__(preprocessor=preprocessor, vocab=expanded_vocab)  # TODO: You obviously want to reduce the vocabulary to the .present tokens. Right now, this is wasteful.

        # Convert to more complex internal data structures
        verbose_vocab = {t: Token(id=i, str=t, present=True) for t,i in expanded_vocab.items()}  # All the other fields are not relevant for the segmenter; e.g., 'left' and 'right' are not needed for merging due to having the merge map, and for splitting we have the SplitResults.
        merge_map, split_map, splits = PickyBPEDeserialiser_SmallFormat._eventsToDataStructures(verbose_vocab, events)

        # To recover whether a token is present or not, it should have been split and not merged again.
        merge_map_concatenated = {verbose_vocab[left.str+right.str]: ids for (left,right), ids in merge_map.items()}
        for token in verbose_vocab.values():
            if token in split_map:
                if token not in merge_map_concatenated or max(split_map[token]) > max(merge_map_concatenated[token]):
                    token.present = False

        # Initialise internals
        self._backend = _PickyBPETokeniserBackend(
            str2token=verbose_vocab,
            id2token=None,
            id2int=None,
            int2id=None,

            merge_map=merge_map,
            split_map=split_map,
            splits=splits,

            events=None
        )

    def tokenise(self, pretoken: str) -> Tokens:
        token_objects = self._backend._encode_word_by_events(pretoken)
        return [obj.str for obj in token_objects]
