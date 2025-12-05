import warnings

from .base import *
from ...util.types import L, Languish

# CONFIGS = {
#     L("English"): setupEnglish(),
#      L("German"): setupGerman(),
#       L("Dutch"): setupDutch()
# }


class DeterministicBPETokeniserWithLanguage(DeterministicBPETokeniser):
    """
    Wrapper around the BPE-knockout tokeniser implementation that not only abstracts away the initialisation config, but
    also the language config.
    """

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab, merges: MergeList,
                 language: Languish,
                 iterations: int, do_knockout: bool, do_reify: bool, backwards_compatible: bool=False):
        language = L(language)

        # Get morphology config
        if language not in CONFIGS:
            warnings.warn(f"Language {language.display_name()} has no BPE-knockout configuration. Defaulting to English.")
        config = CONFIGS.get(language, CONFIGS[L("English")])

        # Run knockout in the context of that language
        with KnockoutDataConfiguration(config):
            super().__init__(
                preprocessor=preprocessor,

                vocab=vocab,
                merges=merges,

                do_morphemic_knockout=do_knockout,
                do_reification=do_reify,
                backwards_compatible=backwards_compatible,
                iterations=iterations
            )


class BPEKnockout(DeterministicBPETokeniserWithLanguage):

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab, merges: MergeList, language: Languish):
        super().__init__(
            preprocessor=preprocessor,

            vocab=vocab,
            merges=merges,

            language=language,

            do_knockout=True,
            do_reify=False,
            iterations=1
        )

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast, specials: AutoVocabSpecs[WithSpecials],
                        language: Languish) -> Self:
        """
        Assuming the given tokeniser is a BPE tokeniser, convert it to a native TkTkT BPE tokeniser
        (rather than wrapping it), and *also* apply knockout using the given language.
        """
        return cls(
            preprocessor=HuggingFacePreprocessor(hf_bpe_tokenizer),

            vocab=AutoVocab.fromTokenizer(hf_bpe_tokenizer, specials),
            merges=HuggingFaceTokeniserPath.fromTokeniser(hf_bpe_tokenizer).loadMerges(),

            language=language
        )


class ReBPE(DeterministicBPETokeniserWithLanguage):

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab, merges: MergeList,
                 language: Languish, iterations: int, backwards_compatible: bool=False):
        super().__init__(
            preprocessor=preprocessor,

            vocab=vocab,
            merges=merges,

            language=language,

            do_knockout=True,
            do_reify=True,
            iterations=iterations,
            backwards_compatible=backwards_compatible
        )

    @classmethod
    def fromHuggingFace(cls, hf_bpe_tokenizer: PreTrainedTokenizerFast, specials: AutoVocabSpecs[WithSpecials],
                        language: Languish, iterations: int, backwards_compatible: bool) -> Self:
        return cls(
            preprocessor=HuggingFacePreprocessor(hf_bpe_tokenizer),

            vocab=AutoVocab.fromTokenizer(hf_bpe_tokenizer, specials),
            merges=HuggingFaceTokeniserPath.fromTokeniser(hf_bpe_tokenizer).loadMerges(),

            language=language,

            iterations=iterations,
            backwards_compatible=backwards_compatible
        )
