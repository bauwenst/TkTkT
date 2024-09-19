"""
Pythonic wrapper around the SentencePiece package's KudoPiece trainer.
Why not use `tokenizers`? Because I trust Kudo himself more than HuggingFace (since they can't even explain it),
and at least he has documentation that exists.
"""
from pathlib import Path
from typing import List, Iterable, Tuple
from dataclasses import dataclass

import sentencepiece
from modest.formats.tsv import iterateTsv

from ...preparation.boundaries import BoundaryMarkerLocation
from ...interfaces.vocabulariser import Vocabulariser, Preprocessor, UnidentifiedVocab
from ...util.timing import datetimeDashed

MAXIMUM_SENTENCE_LENGTH = 4192


@dataclass
class KudoPieceArguments_Alphabet:
    required_chars: List[str]
    byte_fallback: bool
    character_coverage: float=0.9995


@dataclass
class KudoPieceArguments_Algorithm:
    initial_vocab_size: int=1_000_000
    maximum_token_length: int=16
    shrinking_factor: float=0.75
    num_sub_iterations: int=2


class KudoPieceTrainer(Vocabulariser):

    def __init__(self, preprocessor: Preprocessor,
                 final_vocab_size: int, word_boundary_location: BoundaryMarkerLocation,
                 alphabet_arguments: KudoPieceArguments_Alphabet,
                 algorithm_arguments: KudoPieceArguments_Algorithm,
                 file_stem: str="kudopiece"):
        """
        Trainer for KudoPiece (a.k.a. ULM) tokenisers.

        As of writing, SentencePiece does not yet support isolated word boundaries nor a custom symbol; it will always
        be "_". You can decide the location though.

        ---

        Some documentation about documentation.

        The .train method can be seen in the Python implementation to have two special arguments, also demonstrated in
        https://github.com/google/sentencepiece/blob/master/python/README.md#model-training
            --sentence_iterator: train from iterator.
            --model_writer: instead of using cwd to output the model, call .write on this object.

        The most important arguments are explained under
        https://github.com/google/sentencepiece#train-sentencepiece-model
            --input: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor. By default, SentencePiece normalizes the input with Unicode NFKC. You can pass a comma-separated list of files.
            --model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
            --vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
            --character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with rich character set like Japanese or Chinese and 1.0 for other languages with small character set.
            --model_type: model type. Choose from unigram (default), bpe, char, or word. The input sentence must be pretokenized when using word type.

        The full list of arguments is under
        https://github.com/google/sentencepiece/blob/master/doc/options.md
        Of which the following are the most important for us:
            --model_type (model algorithm: unigram, bpe, word or char)  type: std::string default: "unigram"

        I/O
            --input_sentence_size (maximum amount of sentences the trainer loads)  type: std::uint64_t default: 0
            --max_sentence_length (maximum length of sentence in byte; longer are just ignored)  type: int32 default: 4192
            --input_format (Input format. Supported format is `text` or `tsv` (string-tab-count iteration.)  type: std::string default: ""
            --train_extremely_large_corpus (Increase bit depth for unigram tokenization.)  type: bool default: false

            --model_prefix (output model prefix)  type: std::string default: ""

        Alphabet:
            --character_coverage (character coverage to determine the minimum symbols)  type: double default: 0.9995
            --required_chars (UTF8 characters in this flag are always used in the character set regardless of --character_coverage)  type: std::string default: ""
            --byte_fallback (decompose unknown pieces into UTF-8 byte pieces)  type: bool default: false

        There are two lists of predefined tokens: https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md
            --control_symbols (will never be produced when tokenising text)
            --user_defined_symbols (will always be extracted as one token)  type: std::string default: ""

        Initial vocabulary:
            --seed_sentencepiece_size (the size of seed sentencepieces)  type: int32 default: 1000000
            --max_sentencepiece_length (maximum length of sentence piece)  type: int32 default: 16

        Final vocabulary:
            --shrinking_factor (Keeps top shrinking_factor pieces with respect to the loss)  type: double default: 0.75
            --num_sub_iterations (number of EM sub-iterations)  type: int32 default: 2
            --vocab_size (vocabulary size)  type: int32 default: 8000

            --hard_vocab_limit (If set to false, --vocab_size is considered as a soft limit.)  type: bool default: true
            --vocabulary_output_piece_score (Define score in vocab file)  type: bool default: true

            --treat_whitespace_as_suffix (treat whitespace marker as suffix instead of prefix.)  type: bool default: false
            --allow_whitespace_only_pieces (allow pieces that only contain (consecutive) whitespace tokens)  type: bool default: false

        Normalisation (which is cool, but we want to control all of this):
            --normalization_rule_name (Normalization rule name. Choose from nfkc or identity)  type: std::string default: "nmt_nfkc"
            --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool default: true
            --remove_extra_whitespaces (Removes leading, trailing, and duplicate internal whitespace)  type: bool default: true
            --split_by_unicode_script (use Unicode script to split sentence pieces)  type: bool default: true
            --split_by_number (split tokens by numbers (0-9))  type: bool default: true
            --split_by_whitespace (use a white space to split sentence pieces)  type: bool default: true
            --split_digits (split all digits (0-9) into separate pieces)  type: bool default: false
        """
        super().__init__(name="kudopiece", preprocessor=preprocessor)

        if word_boundary_location == BoundaryMarkerLocation.ISOLATED:
            raise ValueError("KudoPiece only supports start-of-word and end-of-word boundary markers.")

        self.alphabet = alphabet_arguments
        self.algorithm = algorithm_arguments
        self.size = final_vocab_size
        self.stem = file_stem

        self.boundary_style = word_boundary_location

    def _vocabulariseFromWords(self, word_iterable: Iterable[Tuple[str,int]]) -> Path:
        """
        FIXME: Currently suffers from https://github.com/google/sentencepiece/issues/967
        """
        return self._withSentencepieceTrainer(
            (f"{self._addSpace(word)}\t{count}" for word, count in word_iterable),
            is_wordfile=True, strings_need_space_splitting=False
        )

    def _vocabulariseFromSentences(self, sentence_iterable: Iterable[str]) -> Path:
        return self._withSentencepieceTrainer(
            map(lambda s: " ".join(self.preprocessor.do(s)), sentence_iterable),
            is_wordfile=False, strings_need_space_splitting=True
        )

    def _withSentencepieceTrainer(self, string_iterable: Iterable[str], is_wordfile: bool=False,
                                  strings_need_space_splitting: bool=False) -> Path:
        output_prefix = self._makeOutputFolder() / (self.stem + "_" + datetimeDashed())

        sentencepiece.SentencePieceTrainer.Train(
            model_type="unigram",

            #  I/O
            sentence_iterator=string_iterable,
            input_format="tsv" if is_wordfile else "",
            max_sentence_length=MAXIMUM_SENTENCE_LENGTH,
            train_extremely_large_corpus=True,  # Why not, right?
            model_prefix=output_prefix.as_posix(),

            # Alphabet
            required_chars=self.alphabet.required_chars,
            byte_fallback=self.alphabet.byte_fallback,
            character_coverage=self.alphabet.character_coverage,

            # Algorithm
            treat_whitespace_as_suffix=self.boundary_style == BoundaryMarkerLocation.END,

            seed_sentencepiece_size=self.algorithm.initial_vocab_size,
            max_sentencepiece_length=self.algorithm.maximum_token_length,
            shrinking_factor=self.algorithm.shrinking_factor,
            num_sub_iterations=self.algorithm.num_sub_iterations,

            vocab_size=self.size,
            hard_vocab_limit=True,
            vocabulary_output_piece_score=True,

            # We assume no special tokens.
            control_symbols=[],
            user_defined_symbols=[],

            # Preprocessing is expected to be done by one of our preprocessors.
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            split_by_unicode_script=False,
            split_by_number=False,
            split_by_whitespace=strings_need_space_splitting,
            split_digits=False,
            allow_whitespace_only_pieces=False  # Ironically, this means that you DO split whitespace into separate pieces. This adheres most to typical behaviour. https://github.com/google/sentencepiece/issues/984
        )

        return output_prefix.with_suffix(".model")

    def _addSpace(self, word: str) -> str:
        return " "*(self.boundary_style == BoundaryMarkerLocation.START) + word + " "*(self.boundary_style == BoundaryMarkerLocation.END)

    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        return [typ for typ,_  in iterateTsv(file_or_folder)]


from bpe_knockout.datahandlers.wordfiles import wordsFileToCounter
def corpusGenerator(wordfile: Path):
    for word, count in wordsFileToCounter(wordfile).items():
        word = " " + word
        words_per_sentence = MAXIMUM_SENTENCE_LENGTH // len(word)
        characters_per_sentence = words_per_sentence*len(word)
        if characters_per_sentence == 0:  # Can't make progress by iterating
            continue

        n_sentences = (count*len(word) - 1) // characters_per_sentence + 1
        for _ in range(n_sentences):
            yield word*words_per_sentence
