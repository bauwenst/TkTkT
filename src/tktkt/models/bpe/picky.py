from typing import Union, Iterable, Tuple
from pathlib import Path
from dataclasses import dataclass

import json
from collections import defaultdict

from pickybpe.utils import MCounter
from pickybpe.vocabularisation import logger, EventType, PickyBPETrainer as _PickyBPEVocabulariserBackend
from pickybpe.segmentation import Token, MergeEventIds, SplitEventIds, SplitResults, PickyBPESegmenter as _PickyBPETokeniserBackend

from ...interfaces import Deserialiser, Vocabulariser, Preprocessor
from ...interfaces.vocabulariser import Vocab
from ...interfaces.tokeniser import TokeniserWithVocabDict, Tokens
from ...util.types import NamedIterable


class _PickyBPEVocabulariserBackend_Extended(_PickyBPEVocabulariserBackend):
    """
    PickyBPE trainer with several additions overtop the original codebase:
        - Supports any boundary marker.
        - Stores the results of training in two much smaller files than the original codebase.

    It already removes support for specials.

    Handling of other input formats, which does not deal with the actual mechanism of PickyBPE, is done by the
    Vocabulariser class that extends this one.
    """

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, picky_threshold: float, character_coverage: float):
        super().__init__(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            picky_threshold=picky_threshold,
            include_specials=False
        )
        self.marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self.marker.intoCharacters(word)

    def _dump(self, file: Union[Path, str]):
        folder = Path(file).resolve()
        if folder.suffix:
            folder = folder.parent
        logger.info(f'Dumping model to {folder.as_posix()}...')

        # Vocabulary just stores types
        with open(folder / "vocab.json", "w", encoding="utf-8") as f:
            json.dump({
                typ.id: typ.to_dict()
                for typ in self.id2token.values()
            }, f, indent=4)

        def validate_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")
            return token

        # Event list is a generalisation of the merge list. "+" means token merge, "-" means token split.
        with open(folder / "events.txt", "w", encoding="utf-8") as f:
            for event_type, start, end in self.events:
                if event_type == EventType.MERGE:
                    f.write(f"+ {' '.join(validate_characters(token.str) for token in start)}\n")
                elif event_type == EventType.SPLIT:
                    f.write(f"- {' '.join(validate_characters(token.str) for token in end)}\n")
                else:
                    raise NotImplementedError

        return folder


@dataclass
class Event:
    left: str
    right: str
    is_merge: bool  # If false, it is a split into these two tokens.


class PickyBPEDeserialiser_SmallFormat(Deserialiser):

    def _verboseVocabularyFromFile(self, path: Path) -> dict[str, Token]:
        folder = Path(path).resolve()
        if folder.suffix:
            folder = folder.parent

        # Get vocabulary
        with open(folder / "vocab.json", "r", encoding="utf-8") as f:
            serialised_vocab = json.load(f)

        token_to_object = dict()
        id_to_object    = dict()
        for token_dict in sorted(serialised_vocab.items(), key=lambda item: item[0]):
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

    def _eventsFromFile(self, path: Path) -> list[Event]:
        folder = Path(path).resolve()
        if folder.suffix:
            folder = folder.parent

        # Get events
        events = []
        with open(folder / "events.txt", "r", encoding="utf-8") as handle:
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


class PickyBPEVocabulariser(Vocabulariser):
    """
    PickyBPE trainer with Vocabulariser interface.
    Supports word frequency files, HuggingFace datasets, ... which were not supported by the original codebase.
    """

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, picky_threshold: float, character_coverage: float):
        super().__init__(name="pickybpe", preprocessor=preprocessor)
        self._backend = _PickyBPEVocabulariserBackend_Extended(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            picky_threshold=picky_threshold,
            character_coverage=character_coverage
        )

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        folder = self._makeOutputFolder(word_iterable.name)
        return self._backend._fit_from_counts(MCounter(dict(self._preprocessWordsToPretokens_counter(word_iterable))), folder, logging_step=100)

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        folder = self._makeOutputFolder(sentence_iterable.name)
        return self._backend._fit_from_counts(MCounter(dict(self._preprocessSentencesToPretokens_counter(sentence_iterable))), folder, logging_step=100)


class PickyBPE(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, events: list[Event]):
        super().__init__(preprocessor=preprocessor, vocab=vocab)

        # Convert to more complex internal data structures
        extended_vocab = {t: Token(id=i, str=t) for t,i in vocab.items()}  # All the other fields are not relevant for the segmenter; e.g., 'left' and 'right' are not needed for merging due to having the merge map, and for splitting we have the SplitResults.
        merge_map, split_map, splits = PickyBPEDeserialiser_SmallFormat._eventsToDataStructures(extended_vocab, events)

        # Initialise internals
        self._backend = _PickyBPETokeniserBackend(
            str2token=extended_vocab,
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
