from abc import abstractmethod
from typing import Union, Iterable, Self
from pathlib import Path
from dataclasses import dataclass

import json
from collections import defaultdict

from pickybpe.vocabularisation import logger, EventType, PickyBPETrainer as _PickyBPETrainerBase
from pickybpe.segmentation import Token, MergeEventIds, SplitEventIds, SplitResults, PickyBPESegmenter as _PickyBPETokeniserBackend

from ...interfaces import Artifacts, CacheableArtifacts
from ...interfaces.vocabularisers import UnidentifiedVocab
from ...interfaces.tokenisers import *
from ...util.iterables import sunion
from .vocabularisation import _VocabulariserWithChizhovBackend, _ChizhovTrainingContext

__all__ = ["PickyBPEVocabulariser", "PickyBPE", "PickyBPEArtifacts"]


@dataclass
class Event:
    tokens: list[str]
    is_merge: bool  # If true, these tokens are concatenated. If false, the concatenation of these tokens is split into them.


class PickyBPEArtifacts(Artifacts):

    @abstractmethod
    def getInternalTypes(self) -> UnidentifiedVocab:
        pass

    @abstractmethod
    def getEvents(self) -> list[Event]:
        pass


class CacheablePickyBPEArtifacts(PickyBPEArtifacts, CacheableArtifacts):

    _NAME_EVENTS = "events.txt"

    def __init__(self, present_types: list[str], all_types: list[str], events: list[Event]):
        super().__init__()
        self._present_types = present_types
        self._all_types     = all_types
        self._events = events

    def _getVocabulary(self) -> UnidentifiedVocab:
        return self._present_types

    def getInternalTypes(self) -> UnidentifiedVocab:
        """
        Returns not only the types in the vocabulary (those that can be produced), but also the intermediate types
        needed for constructing the tokeniser.
        """
        return self._all_types

    def getEvents(self) -> list[Event]:
        return self._events

    def store(self, cache_path: Path):
        folder = Path(cache_path).resolve()
        if not folder.is_dir():
            folder = folder.parent
        logger.info(f'Dumping model to {folder.as_posix()}...')

        # Vocabulary stores types (not verbose types, because we can reconstruct them from merges)
        self._storeTypes(folder, self._present_types)
        self._storeTypes(folder, self._all_types, stem="vocab_internal")

        # Event list is a generalisation of the merge list. "+" means token merge, "-" means token split.
        def validate_merge_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")  # TODO: Probably should not throw an error when writing a vocabulary! We already have a solution for this in _storeTypes.
            return token

        with open(folder / CacheablePickyBPEArtifacts._NAME_EVENTS, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(f"{'+' if event.is_merge else '-'} {' '.join(validate_merge_characters(part) for part in event.tokens)}\n")

        return folder

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return cls._existsTypes(cache_path) and (cache_path / CacheablePickyBPEArtifacts._NAME_EVENTS).exists()

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        cache_path = cache_path.resolve()
        if not cache_path.is_dir():
            cache_path = cache_path.parent

        return CacheablePickyBPEArtifacts(
            present_types=cls._loadTypes(cache_path),
            all_types=cls._loadTypes(cache_path, stem="vocab_internal"),
            events=cls._eventsFromSmallFormat(cache_path / CacheablePickyBPEArtifacts._NAME_EVENTS)
        )

    ####################################################################################################################

    @classmethod
    def _simpleVocabularyFromVerboseFormat(cls, path: Path, only_present: bool=False) -> dict[str, int]:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / _ChizhovBackend_PickyBPE_SmallFormat._NAME_VERBOSE_VOCAB

        with open(path, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)
            return {
                token_dict["str"]: int(token_dict["id"])
                for token_dict in vocab.values() if not token_dict["special"] and (not only_present or token_dict["present"])
            }

    @classmethod
    def _verboseVocabularyFromVerboseFormat(cls, path: Path) -> dict[str, Token]:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / _ChizhovBackend_PickyBPE_SmallFormat._NAME_VERBOSE_VOCAB

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
    def _eventsFromSmallFormat(cls, path: Path) -> list[Event]:
        path = Path(path).resolve()
        if path.is_dir():
            path = path / _ChizhovBackend_PickyBPE_SmallFormat._NAME_EVENTS

        # Get events
        events = []
        with open(path, "r", encoding="utf-8") as handle:
            for event_id, event in enumerate(handle):
                # Parsing
                event = event.strip()
                if not event:
                    continue
                parts = event.split(" ")
                typ, tokens = parts[0], parts[1:]
                if typ not in {"+", "-"}:
                    raise ValueError(f"Encountered line with unknown prefix '{typ}'.")

                # Finally, an object
                events.append(Event(tokens=tokens, is_merge=typ == "+"))

        return events

    @classmethod
    def _eventsToDataStructures(cls, verbose_vocabulary: dict[str, Token], events: list[Event]) -> tuple[MergeEventIds, SplitEventIds, SplitResults]:
        merge_map = defaultdict(list)
        split_map = defaultdict(list)
        splits = dict()

        for event_id, event in enumerate(events):
            if event.is_merge:
                left, right = event.tokens
                merge_map[(verbose_vocabulary[left], verbose_vocabulary[right])].append(event_id)
            else:
                split_map[verbose_vocabulary["".join(event.tokens)]].append(event_id)
                splits[event_id] = [verbose_vocabulary[t] for t in event.tokens]

        return merge_map, split_map, splits


class _ChizhovBackend_PickyBPE_SmallFormat(_PickyBPETrainerBase):
    """
    PickyBPE trainer that stores the results of training in two much smaller files than the original codebase.
    It already removes support for specials.

    Handling of other input formats, which does not deal with the actual mechanism of PickyBPE, is done by the
    Vocabulariser class that extends this one.
    """

    _NAME_VERBOSE_VOCAB = "vocab_verbose.json"
    _NAME_EVENTS = "events.txt"

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, picky_threshold: float, character_coverage: float, max_type_length: int):
        super().__init__(
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            ensured_vocabulary=preprocessor.getAlphabet(),
            max_type_length=max_type_length,
            picky_threshold=picky_threshold,
            include_specials=False
        )
        self._marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self._marker.atomise(word)

    def _dump(self, state: _ChizhovTrainingContext, path: Path):
        folder = Path(path).resolve()
        if not folder.is_dir():
            folder = folder.parent
        logger.info(f'Dumping model to {folder.as_posix()}...')

        # Vocabulary stores verbose types
        with open(folder / _ChizhovBackend_PickyBPE_SmallFormat._NAME_VERBOSE_VOCAB, "w", encoding="utf-8") as f:
            json.dump({
                typ.id: typ.to_dict()
                for typ in sorted(state.str2token.values(), key=lambda token: token.id)
            }, f, indent=4)

        # Event list is a generalisation of the merge list. "+" means token merge, "-" means token split.
        def validate_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")
            return token

        with open(folder / _ChizhovBackend_PickyBPE_SmallFormat._NAME_EVENTS, "w", encoding="utf-8") as f:
            for event_type, start, end in state.events:
                if event_type == EventType.MERGE:
                    f.write(f"+ {' '.join(validate_characters(token.str) for token in start)}\n")
                elif event_type == EventType.SPLIT:
                    f.write(f"- {' '.join(validate_characters(token.str) for token in end)}\n")
                else:
                    raise NotImplementedError

        return folder


class PickyBPEVocabulariser(_VocabulariserWithChizhovBackend[CacheablePickyBPEArtifacts]):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, picky_threshold: float, character_coverage: float, max_type_length: int):
        super().__init__(preprocessor=preprocessor, backend=_ChizhovBackend_PickyBPE_SmallFormat(
            preprocessor=preprocessor,
            vocab_size=vocab_size,  # FIXME: For some reason this number is not quite reached, but we're only like 20 types short. I wonder if this has to do with the same type being formed more than once.
            max_type_length=max_type_length,
            picky_threshold=picky_threshold,
            character_coverage=character_coverage
        ))
        self._threshold = picky_threshold

    def _cacheSubfolder(self) -> str:
        return "pickybpe"

    def _identifierPartial(self) -> str:
        return super()._identifierPartial() + f"_{self._threshold}"

    def _cacheType(self):
        return CacheablePickyBPEArtifacts

    def _dumpToArtifacts(self, dump_path: Path) -> CacheablePickyBPEArtifacts:
        vocab_all     = CacheablePickyBPEArtifacts._simpleVocabularyFromVerboseFormat(dump_path, only_present=False)
        vocab_present = CacheablePickyBPEArtifacts._simpleVocabularyFromVerboseFormat(dump_path, only_present=True)
        events = CacheablePickyBPEArtifacts._eventsFromSmallFormat(dump_path)
        return CacheablePickyBPEArtifacts(
            present_types=sorted(vocab_present.keys(), key=vocab_present.get),
            all_types=sorted(vocab_all.keys(), key=vocab_all.get),
            events=events
        )


########################################################################################################################


class PickyBPE(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, present_vocab: Vocab[WithSpecials], events: list[Event]):
        super().__init__(preprocessor=preprocessor, vocab=present_vocab)

        # Convert to more complex internal data structures.
        # - Step 1: Construct token objects.
        # -- The PickyBPE tokeniser internally also needs some tokens that used to exist but no longer do, to participate
        #    in early events. Luckily, the fact that they are needed for events means we can deduce them from the events.
        all_types = set(present_vocab) | sunion(set(e.tokens) for e in events) | {"".join(e.tokens) for e in events}
        verbose_vocab = {t: Token(id=i, str=t, present=t in present_vocab) for i,t in enumerate(all_types)}  # All the other fields are not relevant for the segmenter; e.g., 'left' and 'right' are not needed for merging due to having the merge map, and for splitting we have the SplitResults.
        merge_map, split_map, splits = CacheablePickyBPEArtifacts._eventsToDataStructures(verbose_vocab, events)

        # - Step 2: To recover whether a token is present or not, it should have been split and not merged again.
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
