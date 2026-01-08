import shutil
from abc import abstractmethod
from typing import Iterator, Iterable, Self
from collections import Counter
from pathlib import Path
from dataclasses import dataclass

import gc

from ...paths import TkTkTPaths
from ...interfaces import Artifacts, CacheableArtifacts
from ...interfaces.vocabularisers import *
from ...util.printing import intsep, percent, pluralise
from ...util.strings import shash

import logging
logger = logging.getLogger(__name__)

_NAME_COUNTS = "words.tsv"
_NAME_CORPUS = "name.txt"


class WordFrequencyList(Artifacts):
    @abstractmethod
    def getFrequencies(self) -> Iterable[tuple[str,int]]:
        pass

    @abstractmethod
    def getCorpusName(self) -> str:
        pass


class CacheableWordFrequencyList(WordFrequencyList, CacheableArtifacts):
    def __init__(self, final_tsv_path: Path, corpus_name: str):  # We save the corpus name only because the result is itself a kind of "corpus". We don't do this for tokenisers.
        super().__init__()
        self._path = final_tsv_path
        self._name = corpus_name

    def _getVocabulary(self) -> UnidentifiedVocab:
        for word, _ in self._readTsv(self._path):
            yield word

    def getFrequencies(self) -> Iterable[tuple[str,int]]:
        yield from self._readTsv(self._path)

    def getCorpusName(self) -> str:
        return self._name

    ####################################################################################################################

    def store(self, cache_path: Path):
        expected_path = cache_path / _NAME_COUNTS
        if self._path != expected_path:
            shutil.copy(self._path, expected_path)  # Or something like _writeTsv(expected, _readTsv(path)) or just move (because the file can be quite big). Moving is allowed because this class is not supposed to be instantiated by a client script, and thus the only time when the path is specified is when it has just been filled by the Vocabulariser.
        with open(cache_path / _NAME_CORPUS, "w", encoding="utf-8") as handle:
            handle.write(self._name + "\n")

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        with open(cache_path / _NAME_CORPUS, "r", encoding="utf-8") as handle:
            name = handle.readline().rstrip()
        return CacheableWordFrequencyList(cache_path / _NAME_COUNTS, name)

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return (cache_path / _NAME_COUNTS).exists()

    ####################################################################################################################

    @classmethod
    def _writeTsv(cls, path: Path, items: Iterable[tuple[str, int]], sort: bool = False):
        NUL = chr(0)
        with open(path, "w", encoding="utf-8") as handle:
            for k, v in (items if not sort else sorted(items, key=lambda t: (-t[1], t[0]))):
                k = k.replace("\t", NUL).replace("\n", NUL)
                handle.write(f"{k}\t{v}\n")

    @classmethod
    def _readTsv(cls, path: Path) -> Iterator[tuple[str, int]]:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                word, count = line.split("\t")
                yield word, int(count)


@dataclass
class CountingConfig:
    checkpoint_every_examples: int
    shard_if_keys_exceed: int         # Upper bound on the amount of keys you want to keep in memory at once when counting. When exceeded, flush/siphon/shard everything to disk and start with an empty counter.
    drop_if_multiple_exceeded: float  # When merging multiple shards, some multiple of the memory of counting is needed. When even this is exceeded, we start dropping low-frequency words (all the 1s, then all the 2s, ...) until we are good again.
    delete_shards_after: bool


class CountWords(UnsupervisedVocabulariser[CacheableWordFrequencyList]):
    """
    Has the goal of compressing a string iterable to a TSV containing every unique word exactly once with its frequency.

    Takes its interface from Vocabulariser to support multiple input formats.
    Has extensive support for caching:
        - Like a Pythagorean siphon, fills up until it reaches a threshold, and then flushes entirely.
          When the counter in memory becomes too big, it is written to disk entirely and reset.
          Afterwards, these saved counters are recombined while filtering.
        - Every so-often, the intermediate results are stored so that even if the counter has not grown too big yet,
          we can resume the counting in case of a crash.
    """

    def __init__(self,
        word_extractor: Preprocessor,
        frequency_minimum: int,
        sort_before_write: bool,
        config: CountingConfig
    ):
        """
        :param frequency_minimum: For filtering the final counter, i.e. with the "true" counts (modulo any dropping that happened while merging), to save space.
        :param sort_before_write: Whether to sort before writing the final counter.
        """
        super().__init__(preprocessor=word_extractor)
        self.frequency_minimum = frequency_minimum
        self.sort_before_write = sort_before_write
        self.config            = config

    def _cacheSubfolder(self) -> str:
        return "counts"

    def _identifierPartial(self) -> str:
        return shash(repr(self.preprocessor)) + "_" + shash(f"{self.config.shard_if_keys_exceed*self.config.drop_if_multiple_exceeded}")

    def _cacheType(self):
        return CacheableWordFrequencyList

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> CacheableWordFrequencyList:
        raise NotImplementedError

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> CacheableWordFrequencyList:
        folder = self._cachePath(sentence_iterable.name)
        folder_intermediate = folder / "shards"
        folder_intermediate.mkdir(exist_ok=True)

        # Find prior work in this folder.
        files = TkTkTPaths.files(folder_intermediate)
        shards      = [file for file in files if "_" not in file.stem]
        checkpoints = [file for file in files if "_"     in file.stem]

        latest_shard      = max(map(lambda path: int(path.stem), shards),                        default=0)
        latest_checkpoint = max(map(lambda path: int(path.stem.removeprefix("_")), checkpoints), default=0)

        if len(checkpoints) > 0 and latest_checkpoint > latest_shard:
            if len(checkpoints) > 1:
                logger.warning(f"Found obsolete checkpoint(s). Only using highest one.")
                for checkpoint in checkpoints:
                    if checkpoint.stem == f"_{latest_checkpoint}":
                        continue
                    checkpoint.unlink()
            counter = self._loadCounter(folder_intermediate / f"_{latest_checkpoint}.tsv")
            resume_after_idx = latest_checkpoint
        else:  # No checkpoint to load. It either doesn't exist or shards are more recent.
            if len(checkpoints) > 0:
                logger.warning(f"Found obsolete checkpoints. Shards are more recent.")
                for checkpoint in checkpoints:
                    checkpoint.unlink()
                latest_checkpoint = 0
            counter = Counter()
            resume_after_idx = latest_shard

        if self._cacheStatusRead(folder) == "merging":
            return CacheableWordFrequencyList(self._merge(shards, output_folder=folder), sentence_iterable.name)

        # Now iterate.
        idx = 0
        for idx,text in enumerate(sentence_iterable, start=1):
            # Resume where you left off.
            if idx <= resume_after_idx:
                continue

            for word in self.preprocessor.do(text):
                counter[word] += 1

            # Flush to disk if counter is too big.
            if len(counter) > self.config.shard_if_keys_exceed:
                shards.append(self._saveCounter(counter, folder_intermediate, n_examples_seen=idx, is_temporary=False))
                counter = Counter()
                gc.collect()

                if latest_checkpoint:
                    self._removeCheckpoint(folder_intermediate, latest_checkpoint)
                latest_checkpoint = 0

            elif idx % self.config.checkpoint_every_examples == 0:
                self._saveCounter(counter, folder_intermediate, n_examples_seen=idx, is_temporary=True)

                if latest_checkpoint:
                    self._removeCheckpoint(folder_intermediate, latest_checkpoint)
                latest_checkpoint = idx

        # For safety, store the current incomplete counter as a shard before continuing to the final merge step.
        if counter:
            shards.append(self._saveCounter(counter, folder_intermediate, idx, is_temporary=False))

            if latest_checkpoint:
                self._removeCheckpoint(folder_intermediate, latest_checkpoint)

        # Merge and delete shards
        self._cacheStatusWrite(folder, "merging")
        return CacheableWordFrequencyList(self._merge(shards, output_folder=folder), sentence_iterable.name)

    def _merge(self, shards: list[Path], output_folder: Path) -> Path:
        """
        Note: if you will only ever stream the counts rather than hold them in memory all at once, this implementation
        is lossier than you need, since it assumes that at the end, the full counter must be held in memory before writing it away.
        """
        warn_size = self.config.shard_if_keys_exceed
        max_size  = int(self.config.shard_if_keys_exceed * self.config.drop_if_multiple_exceeded)
        warned = False

        # Collect
        total_counter = Counter()
        for idx, shard in enumerate(shards, start=1):
            logger.info(f"Reading shard {shard.name}...")
            for word, count in CacheableWordFrequencyList._readTsv(shard):
                total_counter[word] += count

                # Check for overflow
                current_size = len(total_counter)
                if current_size > max_size:
                    logger.warning(f"Merged counter exceeded maximum size of {intsep(max_size)} keys. Trimming.")
                    f = 1
                    while len(total_counter) > max_size:
                        logger.info(f"\tTrimming keys with current frequency {f}...")
                        # Depending on whether you trim a lot or a little from the counter, it is more efficient to store
                        # the words to keep versus storing the words to trim.
                        words_to_trim = {key for key, count in total_counter.items() if count == f}
                        for word in words_to_trim:
                            del total_counter[word]  # I've read that .pop() doesn't allow garbage collection the same way.
                        gc.collect()  # We can't rely on the interpreter to decide when to garbage-collect those del'd items.
                        f += 1
                elif not warned and current_size > warn_size:
                    logger.warning(f"Merged counter has become bigger than during counting ({intsep(warn_size)} keys).")
                    warned = True

            logger.info(f"Keys after {pluralise(idx, 'shard')}: {intsep(len(total_counter))}")

        logger.info(f"Finished merging. Collected {intsep(len(total_counter))} keys totalling {intsep(total_counter.total())} occurrences.")

        if self.frequency_minimum > 0:
            n_keys_before = len(total_counter)
            total_before  = total_counter.total()
            logger.info(f"Filtering frequencies under {self.frequency_minimum}...")

            words_to_trim = {key for key, count in total_counter.items() if count < self.frequency_minimum}
            for word in words_to_trim:
                del total_counter[word]
            gc.collect()

            n_keys_after = len(total_counter)
            total_after  = total_counter.total()
            logger.info(f"Kept {percent(n_keys_after, n_keys_before)} of all keys and {percent(total_after, total_before)} of occurrences.")

        # Save
        output = output_folder / _NAME_COUNTS
        logger.info(f"S{'orting and s' if self.sort_before_write else ''}aving {intsep(len(total_counter))} keys totalling {intsep(total_counter.total())} occurrences...")
        CacheableWordFrequencyList._writeTsv(output, total_counter.items(), sort=self.sort_before_write)

        # Delete
        self._cacheStatusClear(output_folder)
        if self.config.delete_shards_after:
            logger.info("Deleting shards...")
            parents = set()
            for shard in shards:
                parents.add(shard.parent)
                shard.unlink()
            if len(parents) == 1:
                parents.pop().rmdir()

        return output

    def _saveCounter(self, counter: Counter, folder: Path, n_examples_seen: int, is_temporary: bool) -> Path:
        path = folder / ("_"*is_temporary + f"{n_examples_seen}.tsv")
        CacheableWordFrequencyList._writeTsv(path, counter.items())
        return path

    def _loadCounter(self, path: Path) -> Counter:
        counter = Counter()
        for word, count in CacheableWordFrequencyList._readTsv(path):
            counter[word] += count
        return counter

    def _removeCheckpoint(self, folder: Path, checkpoint_id: int):
        path = folder / f"_{checkpoint_id}.tsv"
        path.unlink()
