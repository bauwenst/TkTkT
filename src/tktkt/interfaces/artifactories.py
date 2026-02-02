from typing import TypeVar, Generic, Union, Optional, Callable, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

from .identifiers import WithSpecials, NoSpecials, UnidentifiedVocab, SpecialsExtended, Vocab
from .preprocessors import Preprocessor
from .tokenisers import Tokeniser
from ..util.types import Comparable
from ..util.interfaces import Cacheable
from ..util.strings import anySubstringIn


T = TypeVar("T", bound=Tokeniser)

class TokeniserFactory(ABC, Generic[T]):
    """
    Object that instantiates a tokeniser with mostly default parameters.
    """

    @abstractmethod
    def buildTokeniser(self) -> T:
        pass


_DEFAULT_VOCAB_FILENAME = "vocab.txt"

class Artifacts(ABC):
    """
    Loads a specific instance of vocabularisation results stored on disk somewhere (usually remotely).

      - Vocabularisers represent an algorithm, e.g. BPE.

      - Artifacts represent a file format, e.g. the Sennrich format, and the result of running an algorithm, e.g. the BPE vocab and merges.
        "Result" can mean two things:
            - Any result, even at runtime (CacheableArtifacts);
            - The result of ONE SPECIFIC run in the past, like applying BPE with |V| = 32k to SlimPajama's first 3M examples;
              For such Artifacts, part of the training code has to be re-declared since the specific training script
              is not accessible. Thus, caching is not available for these Artifacts.

      - TokeniserFactories, lastly, are objects that abstract over the constructor of a Tokeniser, often using Artifacts in the
        process, but with customisable parameters. A factory usually has three arguments: a Preprocessor (for what happens before
        tokenisation), Artifacts (for what happens during tokenisation), and Specials (for what happens after tokenisation).
    """

    # Implementational note:
    #
    # For a long time, Artifacts used to be a Generic[WithSpecials]. Although it was nice to deliver everything needed to
    # construct the preprocessor and the vocabulary as one object, it did not really make sense to wrap the Specials in
    # an Artifacts object (passed to it in the constructor) to then just have those Specials be wrapped in the Vocab
    # object that could be asked of it -- both because a "wrapper for Specials" already existed (Vocab) and because it
    # required all implementations of Artifacts to adhere to the standard that self._specials would be given to Vocab.
    #
    # At that time, the return value of running a Vocabulariser was a Path to its computational results on disk. This also
    # put Artifacts in a weird spot, because if the Vocabulariser knew how to store its computations, then it should also
    # have the knowledge to load them, and so it effectively became a computer AND a serialiser/deserialiser whereas the
    # Artifacts didn't even know how to load/store itself. When that return type changed to an Artifacts object rather
    # than Path, not only was load/store responsibility given to Artifacts instead, but it also became clear that Specials
    # couldn't possibly be counted as the result of a Vocabulariser's computations: both because conceptually it makes no
    # sense (the Vocabulariser has nothing to do with the choice of Specials), and also because it created a big mess of
    # generics in Vocabulariser (parent class Vocabulariser(Generic[T_Artifacts, WithSpecials]) with every single method
    # looking like method(self, ..., specials: WithSpecials) -> T_Artifacts, and then subclasses like
    # BPEVocabulariser(Vocabulariser[BPEArtifacts[WithSpecials]], WithSpecials) where the second WithSpecials would link the
    # WithSpecials in the method arguments to the WithSpecials in the method output).
    #
    # Hence, Artifacts lost its generic-ness, and the Specials just became a method argument on Artifacts itself for the
    # moment a Vocab to wrap those Specials is actually required. The only effect for users is that anywhere you would use
    # Artifacts (namely, in a Factory, since it constructs a Vocab without the user's control), you now also need the
    # Specials you would've given to their constructor originally (which didn't even work properly to induce the generic type).

    def __init__(self):
        self._vocab_cache = None

    def getVocabulary(self,
        specials: SpecialsExtended[WithSpecials]=SpecialsExtended(NoSpecials()),
        type_sorting_key: Callable[[str], Comparable]=None
    ) -> Vocab[WithSpecials]:
        """
        Combines the Vocabulariser-specific knowledge of ._getVocabulary() with the abilities to choose
            1. the order of the remaining types;
            2. which specials to add to them.
        """
        if self._vocab_cache is None:
            removed_types = self._bakedSpecials()
            iterator = filter(lambda t: t not in removed_types, self._getVocabulary())
            if type_sorting_key is not None:
                iterator = sorted(iterator, key=type_sorting_key)
            self._vocab_cache = Vocab(iterator, specials=specials.specials, unk_id=specials.unk)
        return self._vocab_cache

    @abstractmethod
    def _getVocabulary(self) -> UnidentifiedVocab:
        """Constructs the vocabulary. The result is cached in this object so that it can be quickly recalled if needed again."""
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


class CacheableArtifacts(Artifacts, Cacheable):
    """
    Artifacts constructed in the same runtime as they are loaded.

    The implication is that all the objects used to construct these Artifacts (e.g. Vocabulariser parameters, Preprocessor, ...)
    all still exist in the runtime, and do not have to be defined by this object, nor serialised (the Python script is that serialised form, really).
    """
    def __init__(self):
        super().__init__()
        self._preprocessor_effective: Optional[Preprocessor] = None
        self._preprocessor_native:    Optional[Preprocessor] = None

    def _bakedSpecials(self) -> set[str]:
        return set()

    def setPreprocessors(self, effective: Preprocessor, native: Optional[Preprocessor]=None):
        self._preprocessor_effective = effective
        self._preprocessor_native    = native or effective

    def preprocessorNative(self) -> Preprocessor:
        assert self._preprocessor_native is not None
        return self._preprocessor_native

    def preprocessorEffective(self) -> Preprocessor:
        assert self._preprocessor_effective is not None
        return self._preprocessor_effective

    # Auxiliary methods for saving vocabularies as a simple txt.

    @classmethod
    def _storeTypes(cls, cache_path: Path, types: Union[dict[str,int], Iterable[str]]):
        if isinstance(types, dict):
            types = sorted(types.keys(), key=types.get)

        # We use two obsolete ASCII control characters, namely byte 5 and byte 6, to encode \r\n, which are control characters that may actually be used below.
        LF = chr(5)
        CR = chr(6)
        with open(cache_path / _DEFAULT_VOCAB_FILENAME, "w", encoding="utf-8") as handle:
            for t in types:
                t = t.replace("\n", LF).replace("\r", CR)
                handle.write(f"{t}\n")

    @classmethod
    def _loadTypes(cls, cache_path: Path) -> list[str]:
        LF = chr(5)
        CR = chr(6)
        with open(cache_path / _DEFAULT_VOCAB_FILENAME, "r", encoding="utf-8") as handle:
            return [line.rstrip("\r\n").replace(LF, "\n").replace(CR, "\r") for line in handle.readlines() if line.rstrip("\r\n")]

    @classmethod
    def _existsTypes(cls, cache_path: Path) -> bool:
        return (cache_path / _DEFAULT_VOCAB_FILENAME).is_file()
