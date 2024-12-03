from typing import Dict, Tuple, Union, List, Iterable
from pathlib import Path
from enum import Enum
from collections import defaultdict, OrderedDict

import json
from tqdm.auto import tqdm

# Core libraries
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import bpeasy
from ..kudopiece.vocabularisation import KudoPieceVocabulariser, EXOTIC_SCRIPT_PRETOKEN_SEPARATOR
import bpe_knockout._lib.sbpe.learn_bpe as sbpe

from bpe_knockout.auxiliary.tokenizer_interface import SennrichTokeniserPath, HuggingFaceTokeniserPath
from modest.formats.tsv import iterateTsv

from ...preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from ...preparation.mappers import PseudoByteMapping
from ...preparation.instances import KudoSpaceMarker
from ...interfaces.vocabulariser import Vocabulariser, Preprocessor, NamedIterable, UnidentifiedVocab, DEFAULT_FIVE_SPECIALS
from ...util.dicts import substituteKey, argmax, kargmax
from ...util.iterables import streamProgress
from ...util.printing import logger, pluralise


Merges = List[Tuple[str,str]]


class BpeTrainerImplementation(Enum):
    SENTENCEPIECE = 1
    HUGGINGFACE = 2
    BPEASY = 3
    SBPE = 4


class BPEVocabulariser(Vocabulariser):

    def __init__(self, preprocessor: Preprocessor,
                 vocab_size: int, skip_sentences_over_length: int=8192, max_token_length: int=64,
                 implementation: BpeTrainerImplementation=BpeTrainerImplementation.SENTENCEPIECE,
                 replace_boundary_marker_with: BoundaryMarker=None):
        super().__init__(name="bpe", preprocessor=preprocessor)
        self._size = vocab_size
        self._max_token_length = max_token_length
        self._max_sentence_length = skip_sentences_over_length
        self._marker = self.preprocessor.getBoundaryMarker()
        self._replacement_marker = replace_boundary_marker_with or self._marker

        assert self._marker.location == self._replacement_marker.location

        self._mode = implementation

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        # Note: The cases that preprocess beforehand do this because the trainers only accept sentences. The other cases do preprocessing internally.
        if self._mode == BpeTrainerImplementation.HUGGINGFACE:
            return self._withHfTrainer(self._preprocessWordsToSentences(word_iterable))
        elif self._mode == BpeTrainerImplementation.BPEASY:
            return self._withBPEasyTrainer(self._preprocessWordsToSentences(word_iterable))
        elif self._mode == BpeTrainerImplementation.SBPE:
            return self._withSBPETrainer(word_iterable, is_wordfile=True)
        elif self._mode == BpeTrainerImplementation.SENTENCEPIECE:
            return self._withSentencePieceTrainer(word_iterable, is_wordfile=True)

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        # Note: calling self._preprocessSentencesToSentences is up to the individual trainer implementations below.
        if self._mode == BpeTrainerImplementation.BPEASY:
            return self._withBPEasyTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.HUGGINGFACE:
            return self._withHfTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.SBPE:
            return self._withSBPETrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.SENTENCEPIECE:
            return self._withSentencePieceTrainer(sentence_iterable)

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        if file_or_folder.is_dir():
            file_or_folder = file_or_folder / "vocab.json"
        with open(file_or_folder, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        return sorted(vocab, key=vocab.get)

    ####################################################################################################################

    def _storeVocab(self, vocab: Union[OrderedDict[str, int], Dict[str, int], Iterable[str]], folder: Path) -> Path:
        if not isinstance(vocab, dict):
            vocab = OrderedDict((t,i) for i,t in enumerate(vocab))

        output_path = folder / "vocab.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle, ensure_ascii=False, indent=4)
        return output_path

    def _storeMerges(self, merges: Iterable[Tuple[str,str]], folder: Path) -> Path:
        output_path = folder / "merges.txt"
        with open(output_path, "w", encoding="utf-8") as handle:
            for left, right in merges:
                handle.write(f"{left} {right}\n")

        return output_path

    def _withBPEasyTrainer(self, sentence_iterable: NamedIterable[str]) -> Path:
        """
        Produces a pseudo-byte vocabulary with boundary marker. Assumes that the vocabulariser's preprocessor is slightly
        different from the tokeniser's preprocessor, namely that the former does not apply a pseudo-byte mapping nor adds
        boundaries other than spaces.
        """
        logger("Note: BPEasy has a byte-based implementation. That means you should use a byte-compatible preprocessor, which doesn't add a boundary marker (unless it's a space) and doesn't apply a pseudo-byte mapping.")
        out_folder = self._makeOutputFolder(sentence_iterable.name)

        # Learn vocabulary.
        PRETOKEN_SEPARATOR = "🂠"  # Normally you can use spaces to separate pretokens, but here we need spaces to act as boundary markers, and while all pretokens are separated, not all pretokens have a boundary marker, hence you can't use spaces.
        PRETOKEN_REGEX     = " ?"*(self._marker.location == BoundaryMarkerLocation.START) + "[^" + PRETOKEN_SEPARATOR + "]+" + " ?"*(self._marker.location == BoundaryMarkerLocation.END)
        bytes_vocab: Dict[bytes,int] = bpeasy.train_bpe(
            iterator=streamProgress(self._preprocessSentencesToSentences(sentence_iterable, sep=PRETOKEN_SEPARATOR)).__iter__(),
            python_regex=PRETOKEN_REGEX,  # This regex is not the regex of what to split on, but the regex of pretokens.
            vocab_size=self._size,
            max_token_length=self._max_token_length
        )

        vocab = self._standardiseBPEasyVocab(bytes_vocab)
        types = sorted(vocab, key=vocab.get)
        self._storeVocab(OrderedDict( (typ, vocab.get(typ)) for typ in types ), out_folder)

        # Induce merges
        merges = BPEVocabulariser.deduceMergesFromVocab(types, self._replacement_marker)
        self._storeMerges(merges, out_folder)

        return out_folder

    def _standardiseBPEasyVocab(self, bytes_vocab: Dict[bytes,int]) -> Dict[str,int]:
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

    def _withSBPETrainer(self, iterable: Union[NamedIterable[Tuple[str,int]], NamedIterable[str]], is_wordfile: bool=False) -> Path:
        out_folder = self._makeOutputFolder(iterable.name)

        paths = SennrichTokeniserPath(folder=out_folder)
        path_vocab, path_merges = paths.getPaths()

        # Learn merges
        iterables = []
        if is_wordfile:
            iterables.append(f"{word} {count}" for word, count in iterable)  # Generator
        else:
            iterables.append(iterable)
        with open(path_merges, "w", encoding="utf-8") as out_handle:
            sbpe.learn_bpe(iterables, out_handle, is_dict=is_wordfile,
                           num_symbols_ori=self._size, total_symbols=True,
                           preprocessor=self.preprocessor, marker=self._marker)

        # Deduce vocab
        alphabet = self.preprocessor.getAlphabet()
        vocab = BPEVocabulariser.deduceVocabFromMerges(path_merges, alphabet.getCharacters() if alphabet else [])
        self._storeVocab(vocab, out_folder)
        return out_folder

    def _withHfTrainer(self, sentence_iterable: NamedIterable[str]) -> Path:
        """
        HuggingFace equivalent. For German: starts out extremely slow
        (giving an ETA of 500 000 hours), but finishes in under 2 hours.
        """
        out_folder = self._makeOutputFolder(sentence_iterable.name)

        # Model: no normaliser (because RobBERT doesn't have one) and no decoder (because training is back-end-only).
        tokeniser = Tokenizer(models.BPE())
        tokeniser.pre_tokenizer = None
        tokeniser.decoder       = None
        # tokeniser.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True)
        # tokeniser.decoder       = decoders.ByteLevel()
        # tokeniser.pre_tokenizer = pre_tokenizers.Metaspace(replacement=self.marker.substitute)
        # tokeniser.decoder       = decoders.Metaspace(replacement=self.marker.substitute)

        # Trainer interface according to https://huggingface.co/docs/tokenizers/api/trainers (ignore the type hints that complain):
        alphabet = self.preprocessor.getAlphabet()
        trainer = trainers.BpeTrainer(
            vocab_size=self._size,
            show_progress=True,
            special_tokens=DEFAULT_FIVE_SPECIALS.all_special_tokens,
            initial_alphabet=alphabet.getCharacters() if alphabet else [],  # after https://huggingface.co/docs/tokenizers/training_from_memory
            max_token_length=self._max_token_length
        )
        tokeniser.train_from_iterator(sentence_iterable, trainer=trainer)

        # Save
        save_path = out_folder / f"tokenizer.json"
        hf = HuggingFaceTokeniserPath(json_path=save_path)  # Calls .mkdir, which is important because otherwise the next line fails.
        tokeniser.save(path=save_path.as_posix())

        # Turn into vocab.json + merges.txt
        vocab, merges = SennrichTokeniserPath(folder=out_folder).getPaths()
        self._storeVocab(hf.loadVocabulary(), out_folder)
        with open(merges, "w", encoding="utf-8") as out_handle:
            out_handle.writelines([merge + "\n" for merge in hf.loadMerges()])
        return out_folder

    def _withSentencePieceTrainer(self, word_or_sentence_iterable: NamedIterable, is_wordfile: bool=False):
        output_prefix = self._makeOutputFolder(word_or_sentence_iterable.name) / "spm"

        if is_wordfile:
            word_or_sentence_iterable = self._preprocessWordsToPretokens_approx(word_or_sentence_iterable)  # TODO: I wonder if SP prefixes every TSV entry with a SoW. If not, you can use the non-approximative _counter variant here too.
        else:
            word_or_sentence_iterable = self._preprocessSentencesToSentences(word_or_sentence_iterable, sep=EXOTIC_SCRIPT_PRETOKEN_SEPARATOR)

        # print(repr(next(iter(iterable))))
        alphabet = self.preprocessor.getAlphabet()
        required_characters = alphabet.getCharacters() if alphabet else []

        KudoPieceVocabulariser._callSentencePieceTrainer(
            actual_vocab_size=self._size,
            model_type="bpe",

            # I/O
            sentence_iterator=streamProgress(word_or_sentence_iterable).__iter__(),
            input_format="tsv" if is_wordfile else "",
            max_sentence_length=self._max_sentence_length,
            train_extremely_large_corpus=True,  # Why not, right?
            model_prefix=output_prefix.as_posix(),

            # Alphabet
            required_chars=[],  # Note: Required characters must have a frequency > 0 otherwise you get an error. Hence, we can only add them in a post-processing step. https://github.com/google/sentencepiece/blob/d8f741853847553169444afc12c00f4bbff3e9ce/src/bpe_model_trainer.cc#L37
            character_coverage=1.0 if required_characters else 0.9995,

            # Algorithm
            treat_whitespace_as_suffix=self._marker.location == BoundaryMarkerLocation.END,

            # seed_sentencepiece_size=self._algorithm.initial_vocab_size,
            max_sentencepiece_length=self._max_token_length,
            # shrinking_factor=self._algorithm.shrinking_factor,
            # num_sub_iterations=self._algorithm.num_sub_iterations,
        )

        # Get vocab
        vocab = self._standardiseSpmVocab(output_prefix.with_suffix(".vocab"), required_characters)
        self._storeVocab(vocab, output_prefix.parent)

        # Deduce merges
        merges = self.deduceMergesFromVocab(sorted(vocab, key=vocab.get), boundary_marker=self._replacement_marker)
        self._storeMerges(merges, output_prefix.parent)

        return output_prefix.parent

    def _standardiseSpmVocab(self, spm_vocab: Path, required_chars: Iterable[str]) -> Dict[str,int]:
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

    ####################################################################################################################

    @classmethod
    def loadMerges(cls, file_or_folder: Path) -> Merges:
        if file_or_folder.is_dir():
            file_or_folder = file_or_folder / "merges.txt"

        with open(file_or_folder, "r", encoding="utf-8") as handle:
            return [tuple(line.strip("\r\n").split(" ")) for line in handle
                    if line.strip("\r\n") and not line.startswith("#version")]

    @staticmethod
    def deduceVocabFromMerges(mergefile: Path, alphabet: List[str]=None) -> Dict[str, int]:
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
        if not alphabet:
            alphabet = used_types - produced_types

        # Combine everything
        vocab = {c: i for i, c in enumerate(
            DEFAULT_FIVE_SPECIALS.all_special_tokens +
            sorted(alphabet) +
            list(produced_types)
        )}

        return vocab

    @staticmethod
    def deduceMergesFromVocab(types: UnidentifiedVocab, boundary_marker: BoundaryMarker) -> List[Tuple[str,str]]:
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
        current_type_state = [list(boundary_marker.intoCharacters(t)) for t in types]
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
