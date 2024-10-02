from typing import Dict, Tuple, Union
from pathlib import Path
from enum import Enum

import json
from tqdm.auto import tqdm

from transformers import SpecialTokensMixin
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import bpeasy
import sentencepiece

from bpe_knockout.auxiliary.tokenizer_interface import SennrichTokeniserPath, HuggingFaceTokeniserPath
from bpe_knockout._lib.sbpe.learn_bpe import learn_bpe

from ...preparation.boundaries import BoundaryMarker
from ...preparation.mappers import PseudoByteMapping
from ...interfaces.vocabulariser import Vocabulariser, Preprocessor, NamedIterable, UnidentifiedVocab


PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"
MSK = "<mask>"
UNK = "<unk>"
SPECIAL_TYPES = SpecialTokensMixin(
    pad_token=PAD,
    bos_token=BOS,
    eos_token=EOS,
    mask_token=MSK,
    unk_token=UNK
)  # The above argument mapping is reconstructed with .special_tokens_map; the list of values is .all_special_tokens


class BpeTrainerImplementation(Enum):
    SBPE   = 1
    HF     = 2
    BPEASY = 3
    SP     = 4


class BPEVocabulariser(Vocabulariser):

    def __init__(self, preprocessor: Preprocessor, boundary_marker: BoundaryMarker, byte_based: bool,
                 vocab_size: int, max_length: int=128,
                 implementation: BpeTrainerImplementation=BpeTrainerImplementation.HF):
        super().__init__(name="bpe", preprocessor=preprocessor)
        self._size   = vocab_size
        self._maxlen = max_length

        self._marker = boundary_marker
        self._byte_based = byte_based

        self._mode = implementation

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        if self._mode == BpeTrainerImplementation.HF:
            return self._withHfTrainer(self._preprocessWordsToSentences(word_iterable))
        if self._mode == BpeTrainerImplementation.BPEASY:
            return self._withBPEasyTrainer(self._preprocessWordsToSentences(word_iterable))
        elif self._mode == BpeTrainerImplementation.SBPE:
            return self._withSBPETrainer(word_iterable, is_wordfile=True)

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        if self._mode == BpeTrainerImplementation.BPEASY:
            return self._withBPEasyTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.HF:
            return self._withHfTrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.SBPE:
            return self._withSBPETrainer(sentence_iterable)
        elif self._mode == BpeTrainerImplementation.SP:
            return self._withSentencePieceTrainer(sentence_iterable)

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        path = file_or_folder / "vocab.json"
        with open(path, "r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        return sorted(vocab, key=vocab.get)

    ####################################################################################################################

    def _withBPEasyTrainer(self, sentence_iterable: NamedIterable[str]) -> Path:
        path = self._makeOutputFolder() / "vocab.json"

        # Learn vocabulary
        vocab = bpeasy.train_bpe(
            iterator=tqdm(self._preprocessSentencesToSentences(sentence_iterable)).__iter__(), python_regex=" ",  # Splitting on spaces will reveal pretokens.
            vocab_size=self._size,
            max_token_length=self._maxlen
        )
        vocab = {"".join(map(PseudoByteMapping.BYTE_TO_PSEUDO.get, byte_sequence)): i
                 for byte_sequence, i in vocab.items()}

        # Write out
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle)
        return path

    def _withSBPETrainer(self, iterable: Union[NamedIterable[Tuple[str,int]], NamedIterable[str]], is_wordfile: bool=False) -> Path:
        out_folder = self._makeOutputFolder()

        paths = SennrichTokeniserPath(folder=out_folder)
        path_vocab, path_merges = paths.getPaths()

        # Learn merges
        with open(path_merges, "w", encoding="utf-8") as out_handle:
            iterables = [iterable if not is_wordfile else (f"{word} {count}" for word,count in iterable)]
            learn_bpe(iterables, out_handle, is_dict=is_wordfile,
                      num_symbols_ori=self._size, total_symbols=True,
                      word_preprocessor=self.preprocessor, marker=self._marker)

        # Deduce vocab
        vocab = BPEVocabulariser.deduceVocabFromMerges(path_merges, byte_based=self._byte_based)
        with open(path_vocab, "w", encoding="utf-8") as out_handle:
            json.dump(vocab, out_handle, ensure_ascii=False, indent=4)
        return out_folder

    def _withHfTrainer(self, sentence_iterable: NamedIterable[str]) -> Path:
        """
        HuggingFace equivalent. For German: starts out extremely slow
        (giving an ETA of 500 000 hours), but finishes in under 2 hours.
        """
        out_folder = self._makeOutputFolder()

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
            special_tokens=SPECIAL_TYPES.all_special_tokens,
            initial_alphabet=list(PseudoByteMapping.PSEUDO_TO_BYTE) if self._byte_based else [],  # after https://huggingface.co/docs/tokenizers/training_from_memory
            max_token_length=self._maxlen
        )
        tokeniser.train_from_iterator(sentence_iterable, trainer=trainer)

        # Save
        save_path = out_folder / f"BPE_from_{sentence_iterable.name}.json"
        hf = HuggingFaceTokeniserPath(json_path=save_path)  # Calls .mkdir, which is important because otherwise the next line fails.
        tokeniser.save(path=save_path.as_posix())

        # Turn into vocab.json + merges.txt
        vocab, merges = SennrichTokeniserPath(folder=out_folder).getPaths()
        with open(vocab, "w", encoding="utf-8") as out_handle:
            json.dump(hf.loadVocabulary(), out_handle, ensure_ascii=False, indent=4)
        with open(merges, "w", encoding="utf-8") as out_handle:
            out_handle.writelines([merge + "\n" for merge in hf.loadMerges()])
        return out_folder

    def _withSentencePieceTrainer(self, sentence_iterable: NamedIterable[str]):  # TODO
        raise NotImplementedError

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
            SPECIAL_TYPES.all_special_tokens +
            sorted(alphabet) +
            list(produced_types)
        )}

        return vocab
