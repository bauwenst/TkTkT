"""
Wrapper around the BPE-knockout tokeniser implementation, to make it easier to work with language configs.
Note that the BTE class itself is already a TkTkT tokeniser, and has more options, but you'll have to know
how to set the language config.
"""
import langcodes
from langcodes import Language
from typing import Union
import warnings

from bpe_knockout.project.config import KnockoutDataConfiguration, setupDutch, setupEnglish, setupGerman
from bpe_knockout.knockout.core import BTE, BteInitConfig, RefMode, ReifyMode, ByteBasedMode

from .base import *


CONFIGS = {
    "en": setupEnglish(),
    "de": setupGerman(),
    "nl": setupDutch()
}


def langstringToLanguage(language: str) -> Language:
    try:
        return langcodes.find(language)  # E.g. "Dutch"
    except:
        try:
            return langcodes.get(language)  # E.g. "nl"
        except:
            raise ValueError(f"Language cannot be recognised: {language}")


class BPEKnockout(BTE):

    def __init__(self, vocab: Vocab, merges: MergeList, language: Union[Language, str],
                 boundary_marker: SpaceMarker, byte_based: bool=True, normaliser: TextMapper=None):
        # Impute language
        if isinstance(language, str):
            language = langstringToLanguage(language)

        # Get morphology config
        if language.to_tag() not in CONFIGS:
            warnings.warn(f"Language {language.display_name()} has no BPE-knockout configuration. Defaulting to English.")
        config = CONFIGS.get(language.to_tag(), CONFIGS.get("en"))

        # Run knockout in the context of that language
        with KnockoutDataConfiguration(config):
            super().__init__(
                BteInitConfig(
                    knockout=RefMode.MORPHEMIC,
                    bytebased=ByteBasedMode.NONE if not byte_based else ByteBasedMode.INPUT_TO_BYTES
                ),
                starting_vocab=vocab, starting_mergelist=merges,
                autorun_modes=True, quiet=True,
                normalisation=normaliser, boundary_marker=boundary_marker
            )
