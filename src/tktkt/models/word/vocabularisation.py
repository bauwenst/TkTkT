from typing import Iterator, Tuple, Iterable
from collections import Counter
from pathlib import Path
from dataclasses import dataclass

import gc

from ...util.printing import intsep, percent, pluralise
from ...util.types import NamedIterable
from ...paths import TkTkTPaths
from ...interfaces import Preprocessor, Vocabulariser
from ...interfaces.vocabulariser import UnidentifiedVocab


import logging
logger = logging.getLogger(__name__)


class CountWords(Vocabulariser):
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

    _RESULT_STEM = "words"

    @dataclass
    class CacheConfig:
        checkpoint_every_examples: int
        flush_if_keys_exceed: int         # Upper bound on the amount of keys you want to keep in memory at once when counting.
        drop_if_multiple_exceeded: float  # When merging multiple caches, some multiple of the memory of counting is needed. When even this is exceeded, we start dropping low-frequency words (all the 1s, then all the 2s, ...) until we are good again.
        delete_cache_after: bool

    def __init__(self,
        word_extractor: Preprocessor,
        frequency_minimum: int,   # For filtering the final counter, i.e. with the "true" counts (modulo any dropping that happened while merging), to save space.)
        sort_before_write: bool,  # Whether to sort before writing the final counter.
        cache_config: "CountWords.CacheConfig"
    ):
        super().__init__(name="counts", preprocessor=word_extractor)
        self.preprocessor = word_extractor
        self.frequency_minimum = frequency_minimum
        self.sort_before_write = sort_before_write
        self.config            = cache_config

    def _makeOutputFolder(self, extra_suffix: str="") -> Path:  # Remove time suffix.
        return TkTkTPaths.extend(TkTkTPaths.pathToEvaluations(), [self._name, self._name + f"_{extra_suffix}"*bool(extra_suffix)])

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        raise NotImplementedError

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        folder = self._makeOutputFolder(sentence_iterable.name)

        file_results = folder / (CountWords._RESULT_STEM + ".tsv")
        if file_results.exists():
            logger.info(f"Returning cached file {file_results.as_posix()}.")
            return file_results

        folder_intermediate = folder / "shards"
        folder_intermediate.mkdir(exist_ok=True)

        # Find prior work in this folder.
        files = TkTkTPaths.files(folder_intermediate)
        caches      = [file for file in files if "_" not in file.stem]
        checkpoints = [file for file in files if "_"     in file.stem]

        latest_cache      = max(map(lambda path: int(path.stem), caches),                        default=0)
        latest_checkpoint = max(map(lambda path: int(path.stem.removeprefix("_")), checkpoints), default=0)

        if len(checkpoints) > 0 and latest_checkpoint > latest_cache:
            if len(checkpoints) > 1:
                logger.warning(f"Found obsolete checkpoint(s). Only using highest one.")
                for checkpoint in checkpoints:
                    if checkpoint.stem == f"_{latest_checkpoint}":
                        continue
                    checkpoint.unlink()
            counter = self._loadCounter(folder_intermediate / f"_{latest_checkpoint}.tsv")
            resume_after_idx = latest_checkpoint
        else:  # No checkpoint to load. It either doesn't exist or caches are more recent.
            if len(checkpoints) > 0:
                logger.warning(f"Found obsolete checkpoints. Caches are more recent.")
                for checkpoint in checkpoints:
                    checkpoint.unlink()
                latest_checkpoint = 0
            counter = Counter()
            resume_after_idx = latest_cache

        done_flag = folder / ".done"
        if done_flag.exists():
            return self._merge(caches, output=file_results)

        # Now iterate.
        idx = 0
        for idx,text in enumerate(sentence_iterable, start=1):
            # Resume where you left off.
            if idx <= resume_after_idx:
                continue

            for word in self.preprocessor.do(text):
                counter[word] += 1

            # Flush to disk if counter is too big.
            if len(counter) > self.config.flush_if_keys_exceed:
                caches.append(self._saveCounter(counter, folder_intermediate, n_examples_seen=idx, is_temporary=False))
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

        # For safety, cache the current incomplete counter before continuing to the final merge step.
        if counter:
            caches.append(self._saveCounter(counter, folder_intermediate, idx, is_temporary=False))

            if latest_checkpoint:
                self._removeCheckpoint(folder_intermediate, latest_checkpoint)

        # Add "done" flag
        with open(done_flag, "w", encoding="utf-8"):
            pass

        # Merge and delete caches
        return self._merge(caches, output=file_results)

    def _merge(self, shards: list[Path], output: Path) -> Path:
        """
        Note: if you will only ever stream the counts rather than hold them in memory all at once, this implementation
        is lossier than you need, since it assumes that at the end, the full counter must be held in memory before writing it away.
        """
        warn_size = self.config.flush_if_keys_exceed
        max_size  = self.config.flush_if_keys_exceed * self.config.checkpoint_every_examples
        warned = False

        # Collect
        total_counter = Counter()
        for idx, shard in enumerate(shards, start=1):
            logger.info(f"Reading shard {shard.name}...")
            for word, count in self._readTsv(shard):
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
                    logger.warning(f"Merged counter has become bigger than during counting ({intsep(max_size)} keys).")
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
        logger.info(f"S{'orting and s' if self.sort_before_write else ''}aving {intsep(len(total_counter))} keys totalling {intsep(total_counter.total())} occurrences...")
        self._writeTsv(output, total_counter.items(), sort=self.sort_before_write)

        # Delete
        if self.config.delete_cache_after:
            logger.info("Deleting caches...")
            parents = set()
            for shard in shards:
                parents.add(shard.parent)
                shard.unlink()
            if len(parents) == 1:
                parents.pop().rmdir()

        return output

    def _saveCounter(self, counter: Counter, folder: Path, n_examples_seen: int, is_temporary: bool) -> Path:
        path = folder / ("_"*is_temporary + f"{n_examples_seen}.tsv")
        self._writeTsv(path, counter.items())
        return path

    def _loadCounter(self, path: Path) -> Counter:
        counter = Counter()
        for word, count in self._readTsv(path):
            counter[word] += count
        return counter

    def _removeCheckpoint(self, folder: Path, checkpoint_id: int):
        path = folder / f"_{checkpoint_id}.tsv"
        path.unlink()

    @classmethod
    def _writeTsv(cls, path: Path, items: Iterable[tuple[str,int]], sort: bool=False):
        NUL = chr(0)
        with open(path, "w", encoding="utf-8") as handle:
            for k,v in (items if not sort else sorted(items, key=lambda t: (-t[1], t[0]))):
                k = k.replace("\t", NUL).replace("\n", NUL)
                handle.write(f"{k}\t{v}\n")

    @classmethod
    def _readTsv(cls, path: Path) -> Iterator[tuple[str,int]]:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\r\n")
                word, count = line.split("\t")
                yield word, int(count)

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        """
        If, for some reason, you want to load the resulting words as if it is a vocabulary.
        """
        if file_or_folder.is_dir():
            file_or_folder = file_or_folder / (CountWords._RESULT_STEM + ".tsv")
        assert file_or_folder.suffix == ".tsv"

        for word, _ in cls._readTsv(file_or_folder):
            yield word
