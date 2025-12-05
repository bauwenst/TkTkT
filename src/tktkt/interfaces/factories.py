from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Union, List, Optional

from .identifiers import WithSpecials, NoSpecials
from .preparation import Preprocessor
from .vocabulariser import Vocab
from .tokeniser import Tokeniser


T = TypeVar("T", bound=Tokeniser)

class TokeniserFactory(Generic[T], ABC):
    """
    Object that instantiates a tokeniser with mostly default parameters.
    """

    @abstractmethod
    def buildTokeniser(self) -> T:
        pass


class Deserialiser(ABC, Generic[WithSpecials]):  # TODO: In a future version, "Artifact" may become the name of this class.
    """
    Loads a specific instance of vocabularisation results stored on disk somewhere (usually remotely).

    Note that it is vocabularisers that already have the knowledge of how to store to disk (and therefore also how to
    load from disk), except when the file format exists somewhere out there but TkTkT doesn't support that training paradigm.

      - Vocabularisers represent an algorithm and file format, e.g. the BPE algorithm and the Sennrich format.

      - Deserialisers represent the result of an algorithm and file content, e.g. the BPE vocab and merges resulting from
        applying BPE with |V| = 32k to SlimPajama's first 3M examples.

      - TokeniserFactories, lastly, are objects that abstract over the constructor of a Tokeniser, often using a Deserialiser in the
        process, but with customisable parameters. Deserialisers have basically no customisability: they load one very specific
        instance. Different results come from different subclasses, not different initialisation.
    """

    def __init__(self, specials: WithSpecials=NoSpecials(), unk_id: Optional[int]=0):
        self._specials: WithSpecials = specials
        self._unk_id = unk_id
        self._vocab_cache: Vocab[WithSpecials] = None

    def buildVocabulary(self) -> Vocab[WithSpecials]:
        if self._vocab_cache is None:
            vocab = self._buildVocabulary()
            for s in self._bakedSpecials():
                if s in vocab:
                    raise RuntimeError(f"At least one of the baked-in specials was not removed from the vocabulary: {s}")
            self._vocab_cache = vocab
        return self._vocab_cache

    @abstractmethod
    def _buildVocabulary(self) -> Vocab[WithSpecials]:
        pass

    @abstractmethod
    def _bakedSpecials(self) -> set[str]:
        """
        Declares the specials that were baked into this vocabulary by accident.
        Since specials should be given at construction, these types should be removed before assigning any IDs.
        """
        pass

    @abstractmethod
    def preprocessorEffective(self) -> Preprocessor:
        """
        The preprocessor that creates pretokens that can be tokenised into tokens of this vocabulary.

        If the vocabulariser had no built-in preprocessor, then this matches the preprocessor used during vocabularisation.
        This is e.g. not the case for packages like SentencePiece where spaces are converted into underscores after the
        TkTkT preprocessor has finished its work.
        """
        pass

    @abstractmethod
    def preprocessorNative(self) -> Preprocessor:
        """
        The preprocessor that creates pretokens usable by the software package that vocabularised this vocabulary and,
        if applicable, can be used for inference with (only) that software package.
        """
        pass
