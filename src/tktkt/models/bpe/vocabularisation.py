from typing import Dict, Tuple, Union, List, Iterable
from pathlib import Path
from enum import Enum
from collections import defaultdict, OrderedDict

import json
from tqdm.auto import tqdm

# Core libraries
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import bpeasy
import sentencepiece
from bpe_knockout._lib.sbpe.learn_bpe import learn_bpe

from bpe_knockout.auxiliary.tokenizer_interface import SennrichTokeniserPath, HuggingFaceTokeniserPath
from modest.formats.tsv import iterateTsv

from ...preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from ...preparation.mappers import PseudoByteMapping
from ...preparation.instances import SentencePiecePreprocessor, KudoSpaceMarker
from ...interfaces.vocabulariser import Vocabulariser, Preprocessor, NamedIterable, UnidentifiedVocab, DEFAULT_FIVE_SPECIALS
from ...util.dicts import substituteKey, argmax, kargmax
from ...util.iterables import streamProgress
from ...util.printing import logger


Merges = List[Tuple[str,str]]


class BpeTrainerImplementation(Enum):
    SBPE        = 1
    HUGGINGFACE = 2
    SENTENCEPIECE = 3
    BPEASY      = 4


class BPEVocabulariser(Vocabulariser):

    def __init__(self, preprocessor: Preprocessor, boundary_marker: BoundaryMarker, byte_based: bool,
                 vocab_size: int, max_length: int=128,
                 implementation: BpeTrainerImplementation=BpeTrainerImplementation.HUGGINGFACE):
        super().__init__(name="bpe", preprocessor=preprocessor)
        self._size   = vocab_size
        self._maxlen = max_length

        self._marker = boundary_marker
        self._byte_based = byte_based

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
        path = file_or_folder / "vocab.json"
        with open(path, "r", encoding="utf-8") as handle:
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
            python_regex=PRETOKEN_REGEX,  # Splitting on spaces will reveal pretokens.
            vocab_size=self._size,
            max_token_length=self._maxlen
        )

        vocab = self._standardiseBPEasyVocab(bytes_vocab)
        types = sorted(vocab, key=vocab.get)
        self._storeVocab(OrderedDict( (typ, vocab.get(typ)) for typ in types ), out_folder)

        # Induce merges
        merges = BPEVocabulariser.deduceMergesFromVocab(types, self._marker)
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
        if self._marker.detached and self._marker.substitute not in vocab:
            last_type = argmax(vocab)[0]
            substituteKey(vocab, last_type, self._marker.substitute)
        else:  # TODO: Technically you should then make room for the entire alphabet with the marker attached, as happens in Sennrich's case.
            pass

        # Replace the space by the vocabulariser's marker.
        vocab = {typ.replace(" ", self._marker.substitute): vocab.get(typ) for typ in vocab}
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
            learn_bpe(iterables, out_handle, is_dict=is_wordfile,
                      num_symbols_ori=self._size, total_symbols=True,
                      preprocessor=self.preprocessor, marker=self._marker)

        # Deduce vocab
        vocab = BPEVocabulariser.deduceVocabFromMerges(path_merges, byte_based=self._byte_based)
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
        trainer = trainers.BpeTrainer(
            vocab_size=self._size,
            show_progress=True,
            special_tokens=DEFAULT_FIVE_SPECIALS.all_special_tokens,
            initial_alphabet=list(PseudoByteMapping.PSEUDO_TO_BYTE) if self._byte_based else [],  # after https://huggingface.co/docs/tokenizers/training_from_memory
            max_token_length=self._maxlen
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

    def _withSentencePieceTrainer(self, iterable: NamedIterable, is_wordfile: bool=False):
        output_prefix = self._makeOutputFolder(iterable.name) / "spm"

        if is_wordfile:
            iterable = self._preprocessWordsToPretokens_approx(iterable)  # TODO: I wonder if SP prefixes every TSV entry with a SoW. If not, you can use the non-approximative _counter variant here too.
        else:
            iterable = self._preprocessSentencesToSentences(iterable)

        sentencepiece.SentencePieceTrainer.Train(
            model_type="bpe",

            # I/O
            sentence_iterator=streamProgress(iterable).__iter__(),
            input_format="tsv" if is_wordfile else "",
            max_sentence_length=8192,
            train_extremely_large_corpus=True,  # Why not, right?
            model_prefix=output_prefix.as_posix(),

            # Alphabet
            required_chars=[],  # Note: Required characters must have a frequency > 0 otherwise you get an error. Hence, we can only add them in a post-processing step. https://github.com/google/sentencepiece/blob/d8f741853847553169444afc12c00f4bbff3e9ce/src/bpe_model_trainer.cc#L37
            # byte_fallback=self._alphabet.byte_fallback,
            character_coverage=1.0 if self._byte_based else 0.9995,

            # Algorithm
            treat_whitespace_as_suffix=self._marker.location == BoundaryMarkerLocation.END,

            # seed_sentencepiece_size=self._algorithm.initial_vocab_size,
            max_sentencepiece_length=self._maxlen,
            # shrinking_factor=self._algorithm.shrinking_factor,
            # num_sub_iterations=self._algorithm.num_sub_iterations,

            vocab_size=self._size,
            hard_vocab_limit=True,
            vocabulary_output_piece_score=True,

            # We assume no special tokens.
            control_symbols=DEFAULT_FIVE_SPECIALS.all_special_tokens,
            user_defined_symbols=[],

            # Preprocessing is expected to be done by one of our preprocessors.
            normalization_rule_name="identity",
            add_dummy_prefix=True,
            remove_extra_whitespaces=False,
            split_by_unicode_script=False,
            split_by_number=False,
            split_by_whitespace=not is_wordfile,
            split_digits=isinstance(self.preprocessor, SentencePiecePreprocessor),
            allow_whitespace_only_pieces=False  # Ironically, this means that you DO split whitespace into separate pieces. This adheres most to typical behaviour. https://github.com/google/sentencepiece/issues/984
        )

        # Get vocab
        vocab = self._standardiseSpmVocab(output_prefix.with_suffix(".vocab"))
        self._storeVocab(vocab, output_prefix.parent)

        # Deduce merges
        merges = self.deduceMergesFromVocab(sorted(vocab, key=vocab.get), boundary_marker=self._marker)
        self._storeMerges(merges, out_folder)

        return output_prefix.parent

    def _standardiseSpmVocab(self, spm_vocab: Path, required_chars: Iterable[str]=None) -> Dict[str,int]:
        if required_chars is None:
            required_chars = [k for k in PseudoByteMapping.PSEUDO_TO_BYTE if k != " "] if self._byte_based else []
        else:
            required_chars = list(required_chars)

        # First, parse the resulting .vocab file and turn it into a str -> int dictionary.
        vocab = OrderedDict()
        for typ, id in iterateTsv(spm_vocab):
            if "-" not in id:  # This is a special token; Kudo doesn't give IDs to it.
                print(f"Found type with uncounted ID ({typ}). Probably a special type. Skipping it.")
                continue

            id = -int(id)
            vocab[typ] = id

        # Second, find all the characters that are yet to be added to the vocab and which you can definitely remove.
        if self._marker.detached and self._marker.substitute not in required_chars:
            required_chars.append(self._marker.substitute)
        missing_required_chars = [c for c in required_chars if c not in vocab]

        # Pop off the top len(required_chars) BPE types (as if training stopped slightly earlier) and substitute them by those characters.
        removed_types = [KudoSpaceMarker.substitute]  # We definitely remove the _ because we are already requiring the marker.
        last_to_first_types = sorted(vocab, key=vocab.get, reverse=True)
        i = 0
        while len(removed_types) < len(missing_required_chars) and i < len(last_to_first_types):
            t = last_to_first_types[i]
            if t not in required_chars and t not in removed_types:  # You can remove it only if we don't need it.
                removed_types.append(t)
            i += 1
        assert len(removed_types) == len(missing_required_chars), f"Vocab size so small it can't even fit the {len(required_chars)} required characters."

        print("Replacing the last BPE types to make room for the required characters:")
        for old, new in zip(reversed(removed_types),missing_required_chars):
            substituteKey(vocab, old, new)
            print("\t", old, "->", new)

        # Thirdly, renumber the entire vocabulary such that the required chars come before anything else.
        vocab = {t:i for i,t in enumerate(required_chars)} | \
                {t:i+len(required_chars) for i,t in enumerate([t for t in vocab if t not in required_chars])}

        # Lastly, replace the Kudo space marker by our own custom space marker.
        vocab = {typ.replace(KudoSpaceMarker.substitute, self._marker.substitute): id for typ,id in vocab.items()}

        return vocab

    ####################################################################################################################

    @classmethod
    def loadMerges(cls, file_or_folder: Path) -> Merges:
        with open(file_or_folder / "merges.txt", "r", encoding="utf-8") as handle:
            return [tuple(line.strip("\r\n").split(" ")) for line in handle
                    if line.strip("\r\n") and not line.startswith("#version")]

    @staticmethod
    def deduceVocabFromMerges(mergefile: Path, byte_based: bool) -> Dict[str, int]:
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
        alphabet = pre_tokenizers.ByteLevel().alphabet() if byte_based else used_types - produced_types

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
        for i in tqdm(range(len(current_type_state)), desc="TYPES DEDUCED"):
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
