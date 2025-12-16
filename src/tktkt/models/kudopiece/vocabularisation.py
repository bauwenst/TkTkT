"""
Pythonic wrapper around the SentencePiece package's KudoPiece trainer.
Why not use `tokenizers`? Because I trust Kudo himself more than HuggingFace (since they can't even explain it),
and at least he has documentation that exists.
"""
from pathlib import Path
from typing import Iterable, Tuple
from dataclasses import dataclass

from modest.formats.tsv import iterateTsv

from ...preparation.boundaries import BoundaryMarkerLocation
from ...interfaces.vocabularisers import *
from ...util.iterables import streamPrint, streamProgress, T


def progress(iterable: Iterable[T]) -> Iterable[T]:
    # return streamProgress(streamPrint(iterable))
    return streamProgress(iterable)


EXOTIC_SCRIPT_PRETOKEN_SEPARATOR = "𐁝"  # We needed a separator that counts as (1) a script different from punctuation/unknown (unlike e.g. 🂠 and ⛡) that (2) won't appear in natural language (unlike e.g. あ). In sentencepiece/data/scripts.txt, there is a separate script class defined for Linear B, an ancient prototypical script. We use one of the Linear B characters whose usage is not understood by archeologists.


@dataclass
class KudoPieceArguments:
    character_coverage: float=0.9995
    skip_sentences_over_length: int=2**13  # 2**13 == 8192

    initial_vocab_size: int=1_000_000
    maximum_token_length: int=64
    shrinking_factor: float=0.75
    num_sub_iterations: int=2


class KudoPieceVocabulariser(UnsupervisedVocabulariser):
    """
    Wrapper around the SentencePiece trainer for KudoPiece.
    This trainer has quite a large memory footprint. Expect every million sentences to consume about 80 GiB of RAM
    (given that the sentences are no longer than 8192 characters).
    """

    def __init__(self, preprocessor: Preprocessor, final_vocab_size: int, arguments: KudoPieceArguments,
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
        import sentencepiece  # assert
        super().__init__(name="kudopiece", preprocessor=preprocessor)

        self._marker = preprocessor.getBoundaryMarker()
        if self._marker.location == BoundaryMarkerLocation.ISOLATED:
            raise ValueError("KudoPiece only supports start-of-word and end-of-word boundary markers.")

        self._arguments = arguments
        self._size = final_vocab_size
        self._stem = file_stem

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        """
        FIXME: Currently suffers from https://github.com/google/sentencepiece/issues/967
        """
        return self._withSentencepieceTrainer(
            word_iterable.map(lambda t: f"{self._addSpace(t[0])}\t{t[1]}"),
            is_wordfile=True
        )

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        return self._withSentencepieceTrainer(
            self._preprocessSentencesToSentences(sentence_iterable, sep=EXOTIC_SCRIPT_PRETOKEN_SEPARATOR),
            is_wordfile=False
        )

    def _withSentencepieceTrainer(self, string_iterable: NamedIterable[str], is_wordfile: bool=False) -> Path:
        output_folder = self._makeOutputFolder(string_iterable.name)

        required_characters = self.preprocessor.getAlphabet()
        KudoPieceVocabulariser._callSentencePieceTrainer(
            actual_vocab_size=self._size,
            model_type="unigram",

            # I/O
            sentence_iterator=progress(string_iterable).__iter__(),
            input_format="tsv" if is_wordfile else "",
            max_sentence_length=self._arguments.skip_sentences_over_length,
            train_extremely_large_corpus=True,  # Why not, right?
            model_prefix=(output_folder / "spm").as_posix(),  # The "spm" will be suffixed with a file extension.

            # Alphabet
            required_chars=required_characters,
            character_coverage=1.0 if required_characters else self._arguments.character_coverage,  # Potentially bad if the required chars are not all in the corpus.

            # Algorithm
            treat_whitespace_as_suffix=self._marker.location == BoundaryMarkerLocation.END,

            seed_sentencepiece_size=self._arguments.initial_vocab_size,
            max_sentencepiece_length=self._arguments.maximum_token_length,
            shrinking_factor=self._arguments.shrinking_factor,
            num_sub_iterations=self._arguments.num_sub_iterations,
        )

        return output_folder

    def _addSpace(self, word: str) -> str:
        return " " * (self._marker.location == BoundaryMarkerLocation.START) + word + " " * (self._marker.location == BoundaryMarkerLocation.END)

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        if file_or_folder.is_dir():
            file_or_folder = file_or_folder / "spm.vocab"
        return [typ for typ,_ in iterateTsv(file_or_folder)]

    @staticmethod
    def _callSentencePieceTrainer(actual_vocab_size: int, **remaining_arguments):
        """
        Calls the SentencePieceTrainer.Train function by passing the given kwargs to it and imputing a bunch of
        standard arguments.
        """
        import sentencepiece as spm
        spm.SentencePieceTrainer.Train(
            **remaining_arguments,

            vocab_size=actual_vocab_size + 1,  # SentencePiece counts specials as belonging to |V| and <unk> is a special you can't turn off.
            hard_vocab_limit=True,
            byte_fallback=False,
            vocabulary_output_piece_score=True,

            # We assume no special tokens. This is because |V| is supposed to be the amount of units with which to represent language, not which exist in the model total.
            control_symbols=[],
            user_defined_symbols=[],
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,

            # Preprocessing is expected to be done by one of our preprocessors.
            normalization_rule_name="identity",
            add_dummy_prefix=False,  # Similar to HF's add_prefix_space. Should not be needed.
            remove_extra_whitespaces=False,
            split_by_whitespace=False,
            split_by_unicode_script=True,  # What this means precisely is "different Unicode scripts cannot be next to each other in a token", where numbers don't count as having any script and can be used as glue. We do this because (1) we need a way to separate pretokens and (2) realistically, nobody would ever want mixed-script tokens. The only downside is that punctuation and letters no longer appear in the same tokens, which is annoying for English contractions.
            split_by_number=True,  # Needed because SentencePiece treats numbers as belonging to any Unicode script, so if you have digit isolation in your preprocessor, you need this so that the pretoken separator (see above) doesn't glue together multiple pretokens. And if you don't have digit isolation, the only effect will be that letter and number sequences can't be in one token, which they shouldn't anyway just like multiple scripts.
            split_digits=False,
            allow_whitespace_only_pieces=False  # Ironically, setting this to False means that you DO split whitespace into separate pieces. This adheres most to typical behaviour. https://github.com/google/sentencepiece/issues/984
        )