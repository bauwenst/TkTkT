from typing import Union, Iterable, Self
from abc import abstractmethod
from pathlib import Path
from enum import Enum
from collections import defaultdict, OrderedDict, Counter

from tqdm.auto import tqdm

from pickybpe.vocabularisation import EventType, BPETrainer as _BPETrainerBase
from bpe_knockout.model.graph import MergeOnDisk
from bpe_knockout.util.storage import HuggingFaceTokeniserPath
from modest.formats.tsv import iterateTsv

from ...preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from ...preparation.mappers import PseudoByteMapping
from ...factories.preprocessors import KudoSpaceMarker
from ...interfaces.artifactories import Artifacts, CacheableArtifacts
from ...interfaces.vocabularisers import *
from ...interfaces.vocabularisers import T_CacheableArtifact
from ...util.dicts import substituteKey, argmax
from ...util.iterables import streamProgress, deduplicate
from ...util.printing import logger, pluralise


class BpeTrainerImplementation(Enum):
    CHIZHOV       = 0
    SENTENCEPIECE = 1
    HUGGINGFACE   = 2
    BPEASY        = 3
    SBPE          = 4


class BPEArtifacts(Artifacts):
    @abstractmethod
    def getMerges(self) -> list[tuple[str,...]]:
        pass


class CacheableBPEArtifacts(CacheableArtifacts, BPEArtifacts):

    def __init__(self, types: list[str], merges: list[tuple[str,...]]):
        super().__init__()
        self._types = types
        self._merges = merges

    def _getVocabulary(self) -> UnidentifiedVocab:
        return self._types

    def getMerges(self) -> list[tuple[str, ...]]:
        return self._merges

    def store(self, cache_path: Path):
        self._storeTypes(cache_path, self._types)
        self._storeMerges(cache_path, self._merges)

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        return cls(types=cls._loadTypes(cache_path), merges=cls._loadMerges(cache_path))

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return cls._existsTypes(cache_path) and (cache_path / "merges.txt").exists()

    def _bakedSpecials(self) -> set[str]:
        return set()

    # @classmethod
    # def _storeVocab(cls, vocab: Union[OrderedDict[str, int], Dict[str, int], Iterable[str]], folder: Path) -> Path:
    #     if not isinstance(vocab, dict):
    #         vocab = OrderedDict((t,i) for i,t in enumerate(vocab))
    #     else:
    #         vocab = OrderedDict((t,vocab[t]) for t in sorted(vocab.keys(), key=vocab.get))
    #
    #     output_path = folder / "vocab.json"
    #     with open(output_path, "w", encoding="utf-8") as handle:
    #         json.dump(vocab, handle, ensure_ascii=False, indent=4)
    #     return output_path

    # @classmethod
    # def _loadVocabulary(cls, file_or_folder: Path) -> UnidentifiedVocab:
    #     if file_or_folder.is_dir():
    #         file_or_folder = file_or_folder / "vocab.json"
    #     with open(file_or_folder, "r", encoding="utf-8") as handle:
    #         vocab = json.load(handle)
    #     return sorted(vocab, key=vocab.get)

    @classmethod
    def _storeMerges(cls, folder: Path, merges: Iterable[MergeOnDisk]) -> Path:
        output_path = folder / "merges.txt"
        with open(output_path, "w", encoding="utf-8") as handle:
            for parts in merges:
                if isinstance(parts, str):
                    parts = parts.split(" ")
                handle.write(f"{' '.join(parts)}\n")

        return output_path

    @classmethod
    def _loadMerges(cls, file_or_folder: Path) -> list[tuple[str,...]]:
        if file_or_folder.is_dir():
            file_or_folder = file_or_folder / "merges.txt"

        with open(file_or_folder, "r", encoding="utf-8") as handle:
            return [tuple(line.strip("\r\n").split(" ")) for line in handle
                    if line.strip("\r\n") and not line.startswith("#version")]


class BPEVocabulariser(UnsupervisedVocabulariser[CacheableBPEArtifacts]):
    """
    Create a BPE tokeniser from a corpus.

    Not all constructor arguments apply to each implementation. (Yes, perhaps these implementations should be subclasses,
    rather than switching using an enumeration.)
    """

    def __init__(self, preprocessor: Preprocessor,
                 vocab_size: int, implementation: BpeTrainerImplementation=BpeTrainerImplementation.CHIZHOV,
                 max_token_length: int=64, character_coverage: float=1.0, sentencepiece_skip_sentences_over_length: int=8192,
                 replace_boundary_marker_with: BoundaryMarker=None):
        """
        :param replace_boundary_marker_with: only applies to implementations that aren't free to use any preprocessor with
                                             any boundary marker (SentencePiece and BPEasy).
        """
        super().__init__(preprocessor=preprocessor)
        self._size = vocab_size
        self._max_token_length = max_token_length

        self._character_coverage = character_coverage
        self._max_sentence_length = sentencepiece_skip_sentences_over_length

        self._marker = self.preprocessor.getBoundaryMarker()
        self._replacement_marker = replace_boundary_marker_with or self._marker
        assert self._marker.location == self._replacement_marker.location

        self._mode = implementation

    def _identifier(self) -> str:
        return "bpe"

    def _cacheType(self):
        return CacheableBPEArtifacts

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> CacheableBPEArtifacts:
        # Note: The cases that preprocess beforehand do this because the trainers only accept sentences. The other cases do preprocessing internally.
        if self._mode == BpeTrainerImplementation.HUGGINGFACE:
            return self._withHfTrainer(self._preprocessWordsToSentences(word_iterable))
        elif self._mode == BpeTrainerImplementation.BPEASY:
            return self._withBPEasyTrainer(self._preprocessWordsToSentences(word_iterable))
        elif self._mode == BpeTrainerImplementation.SBPE:
            return self._withSBPETrainer(word_iterable, words_not_sentences=True)
        elif self._mode == BpeTrainerImplementation.SENTENCEPIECE:
            return self._withSentencePieceTrainer(word_iterable, words_not_sentences=True)
        elif self._mode == BpeTrainerImplementation.CHIZHOV:
            return self._withChizhovTrainer(word_iterable, words_not_sentences=True)
        else:
            raise NotImplementedError()

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> CacheableBPEArtifacts:
        # Note: calling self._preprocessSentencesToSentences is up to the individual trainer implementations below.
        if self._mode == BpeTrainerImplementation.BPEASY:
            return self._withBPEasyTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.HUGGINGFACE:
            return self._withHfTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.SBPE:
            return self._withSBPETrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.SENTENCEPIECE:
            return self._withSentencePieceTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.CHIZHOV:
            return self._withChizhovTrainer(sentence_iterable)
        else:
            raise NotImplementedError()

    def _withBPEasyTrainer(self, sentence_iterable: NamedIterable[str]) -> CacheableBPEArtifacts:
        """
        Produces a pseudo-byte vocabulary with boundary marker. Assumes that the vocabulariser's preprocessor is slightly
        different from the tokeniser's preprocessor, namely that the former does not apply a pseudo-byte mapping nor adds
        boundaries other than spaces.
        """
        import bpeasy
        logger("Note: BPEasy has a byte-based implementation. That means you should use a byte-compatible preprocessor, which doesn't add a boundary marker (unless it's a space) and doesn't apply a pseudo-byte mapping.")

        # Learn vocabulary.
        PRETOKEN_SEPARATOR = "🂠"  # Normally you can use spaces to separate pretokens, but here we need spaces to act as boundary markers, and while all pretokens are separated, not all pretokens have a boundary marker, hence you can't use spaces.
        PRETOKEN_REGEX     = " ?"*(self._marker.location == BoundaryMarkerLocation.START) + "[^" + PRETOKEN_SEPARATOR + "]+" + " ?"*(self._marker.location == BoundaryMarkerLocation.END)
        bytes_vocab: dict[bytes,int] = bpeasy.train_bpe(
            iterator=streamProgress(self._preprocessSentencesToSentences(sentence_iterable, sep=PRETOKEN_SEPARATOR)).__iter__(),
            python_regex=PRETOKEN_REGEX,  # This regex is not the regex of what to split on, but the regex of pretokens.
            vocab_size=self._size,
            max_token_length=self._max_token_length
        )

        vocab = self._standardiseBPEasyVocab(bytes_vocab)
        types = sorted(vocab, key=vocab.get)
        merges = BPEVocabulariser.deduceMergesFromVocab(types, self._replacement_marker)

        return CacheableBPEArtifacts(types=types, merges=merges)

    def _standardiseBPEasyVocab(self, bytes_vocab: dict[bytes,int]) -> dict[str,int]:
        # Convert the byte-level vocabulary to pseudo-byte characters so that it can be written to a text file.
        # The byte for spaces is converted to a space because of the assumption that the preprocessor wipes all spaces
        # except those it wants as a boundary marker.
        SPACE = " "
        SPACE_BYTE = SPACE.encode("utf-8")[0]
        SPACE_PSEUDO = PseudoByteMapping.BYTE_TO_PSEUDO.get(SPACE_BYTE)
        vocab = {"".join(map(PseudoByteMapping.BYTE_TO_PSEUDO.get, byte_sequence)).replace(SPACE_PSEUDO, SPACE): i
                 for byte_sequence, i in bytes_vocab.items()}
        substituteKey(vocab, SPACE, SPACE_PSEUDO)  # The byte you need for encoding actual spaces needs to be kept.

        # Need to make room for the marker itself as a 257th atom (kind of like a special).
        if self._replacement_marker.detached and self._replacement_marker.substitute not in vocab:
            last_type = argmax(vocab)[0]
            substituteKey(vocab, last_type, self._replacement_marker.substitute)
        else:  # TODO: Technically you should then make room for the entire alphabet with the marker attached, as happens in Sennrich's case.
            pass

        # Replace the space by the vocabulariser's marker. Note: the standalone space doesn't exist in the vocab due to the above substituteKey().
        vocab = {typ.replace(" ", self._replacement_marker.substitute): vocab.get(typ) for typ in vocab}
        return vocab

    def _withSBPETrainer(self, iterable: Union[NamedIterable[tuple[str,int]], NamedIterable[str]], words_not_sentences: bool=False) -> CacheableBPEArtifacts:
        import bpe_knockout._lib.sbpe.learn_bpe as sbpe

        # Learn merges
        iterables = []
        if words_not_sentences:
            iterables.append(f"{word} {count}" for word, count in iterable)  # Generator
        else:
            iterables.append(iterable)

        stream_merges_to = self._cachePath(iterable.name) / "temp.txt"
        with open(stream_merges_to, "w", encoding="utf-8") as out_handle:
            sbpe.learn_bpe(iterables, out_handle, is_dict=words_not_sentences,
                           num_symbols_ori=self._size, total_symbols=True,
                           preprocessor=self.preprocessor, marker=self._marker)
        merges = CacheableBPEArtifacts._loadMerges(stream_merges_to)

        # Deduce vocab
        vocab = BPEVocabulariser.deduceVocabFromMerges(stream_merges_to, partial_alphabet=self.preprocessor.getAlphabet())
        return CacheableBPEArtifacts(types=sorted(vocab), merges=merges)

    def _withHfTrainer(self, sentence_iterable: NamedIterable[str]) -> CacheableBPEArtifacts:
        """
        HuggingFace equivalent. For German: starts out extremely slow
        (giving an ETA of 500 000 hours), but finishes in under 2 hours.
        """
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        out_folder = self._cachePath(sentence_iterable.name)

        # Model: no normaliser (because RobBERT doesn't have one) and no decoder (because training is back-end-only).
        tokeniser = Tokenizer(models.BPE())
        tokeniser.pre_tokenizer = None
        tokeniser.decoder       = None
        # tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
        # tokeniser.decoder       = decoders.ByteLevel()
        # tokeniser.pre_tokenizer = pre_tokenizers.Metaspace(replacement=self.marker.substitute)
        # tokeniser.decoder       = decoders.Metaspace(replacement=self.marker.substitute)

        # Trainer interface according to https://huggingface.co/docs/tokenizers/api/trainers (ignore the type hints that complain):
        trainer = trainers.BpeTrainer(
            vocab_size=self._size,
            show_progress=True,
            special_tokens=[],
            initial_alphabet=self.preprocessor.getAlphabet(),  # after https://huggingface.co/docs/tokenizers/training_from_memory
            max_token_length=self._max_token_length
        )
        tokeniser.train_from_iterator(sentence_iterable, trainer=trainer)

        # Serialise
        save_path = out_folder / f"tokenizer.json"
        hf = HuggingFaceTokeniserPath(json_path=save_path)  # Calls .mkdir, which is important because otherwise the next line fails.
        tokeniser.save(path=save_path.as_posix())

        # Deserialise
        vocab  = hf.loadVocabulary()
        merges = [tuple(m.split(" ")) for m in hf.loadMerges()]
        return CacheableBPEArtifacts(types=sorted(vocab, key=vocab.get), merges=merges)

    def _withSentencePieceTrainer(self, word_or_sentence_iterable: NamedIterable, words_not_sentences: bool=False) -> CacheableBPEArtifacts:
        """
        Benchmarks:
            - Training from a HuggingFace dataset, it takes about 2 hours to stream the next 1 million lines.
            - For 3M space-concatenated sentences and max token length 16, it takes about 30 minutes to train.
            - For 5M exotic-concatenated sentences and max token length 32, it takes at least 769 GiB of RAM and over 8 hours to train.
        """
        from ..kudopiece.vocabularisation import KudoPieceVocabulariser, EXOTIC_SCRIPT_PRETOKEN_SEPARATOR
        output_prefix = self._cachePath(word_or_sentence_iterable.name) / "spm"

        if words_not_sentences:
            word_or_sentence_iterable = self._preprocessWordsToPretokenCounts_approx(word_or_sentence_iterable)  # TODO: I wonder if SP prefixes every TSV entry with a SoW. If not, you can use the non-approximative _counter variant here too.
        else:
            word_or_sentence_iterable = self._preprocessSentencesToSentences(word_or_sentence_iterable, sep=EXOTIC_SCRIPT_PRETOKEN_SEPARATOR)

        # print(repr(next(iter(iterable))))
        KudoPieceVocabulariser._callSentencePieceTrainer(
            actual_vocab_size=self._size,
            model_type="bpe",

            # I/O
            sentence_iterator=streamProgress(word_or_sentence_iterable).__iter__(),
            input_format="tsv" if words_not_sentences else "",
            max_sentence_length=self._max_sentence_length,
            train_extremely_large_corpus=True,  # Why not, right?
            model_prefix=output_prefix.as_posix(),

            # Alphabet
            required_chars=[],  # Note: Required characters must have a frequency > 0 otherwise you get an error. Hence, we can only add them in a post-processing step. https://github.com/google/sentencepiece/blob/d8f741853847553169444afc12c00f4bbff3e9ce/src/bpe_model_trainer.cc#L37
            character_coverage=self._character_coverage,

            # Algorithm
            treat_whitespace_as_suffix=self._marker.location == BoundaryMarkerLocation.END,

            # seed_sentencepiece_size=self._algorithm.initial_vocab_size,
            max_sentencepiece_length=self._max_token_length,
            # shrinking_factor=self._algorithm.shrinking_factor,
            # num_sub_iterations=self._algorithm.num_sub_iterations,
        )

        vocab = self._standardiseSpmVocab(output_prefix.with_suffix(".vocab"), required_chars=self.preprocessor.getAlphabet())
        merges = self.deduceMergesFromVocab(sorted(vocab, key=vocab.get), boundary_marker=self._replacement_marker)
        return CacheableBPEArtifacts(types=sorted(vocab, key=vocab.get), merges=merges)

    def _standardiseSpmVocab(self, spm_vocab: Path, required_chars: Iterable[str]) -> dict[str,int]:
        from ..kudopiece.vocabularisation import EXOTIC_SCRIPT_PRETOKEN_SEPARATOR
        required_chars = list(required_chars)

        # First, parse the resulting .vocab file and turn it into a str -> int dictionary.
        vocab = OrderedDict()
        for typ, id in iterateTsv(spm_vocab):
            if "-" not in id:  # This is a special token; Kudo doesn't give IDs to it.
                print(f"Found type with uncounted ID ({typ}). Probably a special type. Skipping it.")
                continue

            id = -int(id)
            vocab[typ] = id

        print(f"SentencePiece produced {len(vocab)} types.")

        # Second, find all the characters that are yet to be added to the vocab and which you can definitely remove.
        if self._replacement_marker.detached and self._replacement_marker.substitute not in required_chars:
            required_chars.append(self._replacement_marker.substitute)

        # - To be added:
        missing_required_chars = [c for c in required_chars if c not in vocab]

        # - To be removed:
        removed_types = [KudoSpaceMarker.substitute] + [t for t in vocab if EXOTIC_SCRIPT_PRETOKEN_SEPARATOR in t]  # We definitely remove the _ because we are already requiring the marker.
        removed_types = [t for t in removed_types if t in vocab and t not in required_chars]  # Don't delete required characters or things that can't be deleted.

        # - We now have a dual goal: we want to add some types without going over budget, and we want to remove some types.
        #   If we remove more than we want to add, then we don't need to make any more room in the budget. Otherwise, we
        #   pop off the top BPE types (as if training stopped slightly earlier).
        last_to_first_types = sorted(vocab, key=vocab.get, reverse=True)
        i = 0
        while len(removed_types) < len(missing_required_chars) and i < len(last_to_first_types):
            t = last_to_first_types[i]
            if t not in required_chars and t not in removed_types:  # You can remove it when it's not required, and you also can't remove a type twice.
                removed_types.append(t)
            i += 1

        # - We now know for sure which types will free up budget and which types will be added.
        assert len(set(removed_types) & set(missing_required_chars)) == 0
        assert len(removed_types) >= len(missing_required_chars), f"Vocab size so small it can't even fit the {pluralise(len(required_chars), 'required character')}."

        n_excess_types = len(removed_types) - len(missing_required_chars)
        if n_excess_types > 0:  # Excess removal.
            print("Deleting excess from the vocabulary.")
            for t in removed_types[:n_excess_types]:
                print("\tRemoving", t)
                vocab.pop(t)
            removed_types = removed_types[n_excess_types:]

        assert len(removed_types) == len(missing_required_chars)

        if missing_required_chars:
            print("Replacing the last BPE types to make room for the missing required characters:")
            for old, new in zip(removed_types,missing_required_chars):
                substituteKey(vocab, old, new)
                print("\t", old, "->", new)

        print("Finished vocabularising", len(vocab), "types.")

        # Thirdly, renumber the entire vocabulary such that the required chars come before anything else.
        vocab = {t:i for i,t in enumerate(required_chars)} | \
                {t:i+len(required_chars) for i,t in enumerate([t for t in vocab if t not in required_chars])}

        # Lastly, replace the Kudo space marker by our own custom space marker.
        vocab = {typ.replace(KudoSpaceMarker.substitute, self._replacement_marker.substitute): id for typ,id in vocab.items()}

        return vocab

    def _withChizhovTrainer(self, iterable: NamedIterable, words_not_sentences: bool=False) -> CacheableBPEArtifacts:
        # We define a vocabulariser on-the-fly to deal with preprocessing.
        class BPEVocabulariser_Chizhov(_VocabulariserWithChizhovBackend[CacheableBPEArtifacts]):
            def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
                super().__init__(preprocessor=preprocessor, backend=_ChizhovBackend_BPE(
                    preprocessor=preprocessor,
                    vocab_size=vocab_size,
                    character_coverage=character_coverage,
                    max_type_length=max_type_length
                ))

            def _identifier(self) -> str:
                return "bpe"

            def _cacheType(self):
                return CacheableBPEArtifacts

            def _dumpToArtifacts(self, dump_path: Path) -> CacheableBPEArtifacts:
                return CacheableBPEArtifacts(
                    types=CacheableBPEArtifacts._loadTypes(dump_path),
                    merges=CacheableBPEArtifacts._loadMerges(dump_path)
                )

        vocabulariser = BPEVocabulariser_Chizhov(
            preprocessor=self.preprocessor,
            vocab_size=self._size,
            character_coverage=self._character_coverage,
            max_type_length=self._max_token_length
        )
        if words_not_sentences:
            return vocabulariser._vocabulariseFromWords(iterable)
        else:
            return vocabulariser._vocabulariseFromSentences(iterable)

    ####################################################################################################################

    @staticmethod
    def deduceVocabFromMerges(mergefile: Path, partial_alphabet: list[str]=None) -> dict[str, int]:
        # Summarise merges
        with open(mergefile, "r", encoding="utf-8") as in_handle:
            merges = [line.strip() for line in in_handle if line != "#version: 0.2\n"]

        used_types     = set()
        produced_types = set()
        for merge in merges:
            parts = merge.split()
            used_types.update(parts)
            produced_types.add("".join(parts))

        # Get alphabet
        if not partial_alphabet:
            partial_alphabet = []
        alphabet = list(deduplicate(partial_alphabet + sorted(used_types - produced_types)))

        # Combine everything
        vocab = {c: i for i, c in enumerate(
            alphabet +
            list(produced_types)
        )}

        return vocab

    @staticmethod
    def deduceMergesFromVocab(types: UnidentifiedVocab, boundary_marker: BoundaryMarker) -> list[tuple[str,str]]:
        """
        If the types are given in the order they were learnt by the BPE vocabulariser, you can actually reconstruct
        the merge of type i by constructing the BPE tokeniser for types 1 ... i-1 and then tokenising type i with it.
        The result will have exactly two types, and it is those two types that merge into type i (rather than any other
        pair in the vocabulary that also concatenate into type i).

        The theoretical invariant this is based on is that <i>at every point during BPE tokenisation, if there exist
        two adjacent tokens in the sequence that can be merged into a type of the vocabulary, then those tokens are
        exactly the tokens that make up the one merge of that type</i> together with the fact that you don't need to
        know more than the first i-1 merges to simulate a BPE tokeniser with >i merges in its first i-1 inference steps.
        """
        types = list(types)
        current_type_state = [list(boundary_marker.atomise(t)) for t in types]
        states_with_type = defaultdict(set)
        for i in range(len(current_type_state)):
            for atom in current_type_state[i]:
                states_with_type[atom].add(i)

        merges = []
        for i in tqdm(range(len(current_type_state)), desc="Looking for merges in types"):
            tokens_to_merge = tuple(current_type_state[i])
            if len(tokens_to_merge) == 1:  # Alphabet
                continue
            elif len(tokens_to_merge) == 2:  # Concatenate
                left, right = tokens_to_merge
                merged_type = left + right
                assert types[i] == merged_type
                merges.append((left,right))
            else:
                raise RuntimeError("Found type that could not be formed with a binary merge.")

            for j in states_with_type[left] | states_with_type[right]:
                old_tokens = current_type_state[j]
                new_tokens = []
                k = 0
                while k < len(old_tokens):
                    if k != len(old_tokens)-1 and old_tokens[k] == left and old_tokens[k+1] == right:
                        new_tokens.append(merged_type)
                        k += 2
                    else:
                        new_tokens.append(old_tokens[k])
                        k += 1

                if len(old_tokens) != len(new_tokens):  # Update cache.
                    for t in old_tokens:
                        try:
                            states_with_type[t].remove(j)
                        except:
                            pass

                    for t in new_tokens:
                        states_with_type[t].add(j)

                current_type_state[j] = new_tokens

        return merges


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
            ensured_vocabulary=preprocessor.getAlphabet(),
            max_type_length=max_type_length,
            include_specials=False
        )
        self._marker = preprocessor.getBoundaryMarker()

    def _string_to_atoms(self, word: str) -> Iterable[str]:
        return self._marker.atomise(word)

    def _dump(self, path: Path) -> Path:
        from pickybpe.vocabularisation import logger

        folder = Path(path).resolve()
        if folder.suffix:
            folder = folder.parent
        logger.info(f'Dumping model to {folder.as_posix()}...')

        # Vocab
        CacheableBPEArtifacts._storeTypes(folder,
            [typ.str for typ in sorted(filter(lambda token: token != self.unk_token, self.str2token.values()), key=lambda token: token.id)]
        )

        # Merges
        def validate_characters(token: str) -> str:
            for c in token:
                if c.isspace() or len(c.__repr__()) == 6:  # Cannot be printed properly in a text file.
                    raise ValueError(f"Token contains invalid character: {repr(c)}.")
            return token

        CacheableBPEArtifacts._storeMerges(folder,
            (
                [validate_characters(part.str) for part in parts]
                for event_type, parts, _ in self.events if event_type == EventType.MERGE
            )
        )
        return folder


class _VocabulariserWithChizhovBackend(UnsupervisedVocabulariser[T_CacheableArtifact]):
    """
    Puts a Vocabulariser interface around the pickybpe package.
    Supports word frequency files, HuggingFace datasets, ... which were not supported by the original codebase.

    Since the pickybpe package does not depend on TkTkT, its trainer doesn't output Artifacts but a path. This path
    will have to be converted to the relevant Artifacts by subclasses of this class, depending on the format written
    by the trainer.
    """

    def __init__(self, preprocessor: Preprocessor, backend: _BPETrainerBase):
        super().__init__(preprocessor=preprocessor)
        self._backend = backend

    @abstractmethod
    def _dumpToArtifacts(self, dump_path: Path) -> T_CacheableArtifact:
        pass

    def _vocabulariseFromPretokenCounts(self, pretoken_iterable: NamedIterable[tuple[str,int]]) -> T_CacheableArtifact:
        """Does no preprocessing."""
        dump_path = self._backend._fit_from_counts(Counter(dict(pretoken_iterable)), self._cachePath(pretoken_iterable.name), logging_step=100)
        return self._dumpToArtifacts(dump_path)

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[tuple[str,int]]) -> T_CacheableArtifact:
        return self._vocabulariseFromPretokenCounts(self._preprocessWordsToPretokenCounts(word_iterable))

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> T_CacheableArtifact:
        return self._vocabulariseFromPretokenCounts(self._preprocessSentencesToPretokenCounts(sentence_iterable))
