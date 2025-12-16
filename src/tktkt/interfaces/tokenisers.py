from typing import Iterable, Generic
from abc import ABC, abstractmethod

from .identifiers import Vocab, WithSpecials
from .preprocessors import Preprocessor
from ..util.exceptions import MissingUnkError
from ..util.types import Tokens


__all__ = ["Preprocessor", "Tokeniser", "TokeniserWithVocabulary", "Vocab", "WithSpecials", "Tokens"]


class Tokeniser(ABC):
    """
    Subword tokeniser, i.e. an algorithm that turns a string into a list of one or more strings.
    """

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    @abstractmethod
    def tokenise(self, pretoken: str) -> Tokens:
        pass

    def prepareAndTokenise(self, text: str) -> Tokens:  # TODO: Should rename this to preprocessThenTokenise() or tokeniseAfterPreprocessing(), but I will keep this for a 2026 version.
        tokens = []
        for pretoken in self.preprocessor.do(text):
            tokens.extend(self.tokenise(pretoken))
        return tokens

    def getName(self) -> str:  # Can be overridden if your name varies depending on the configuration.
        return self.__class__.__name__

    def __repr__(self):
        return self.getName()


class TokeniserWithVocabulary(Tokeniser, Generic[WithSpecials]):
    """
    Subword tokeniser which additionally contains a bijection from strings it produces to integer identifiers
    (except for a default UNK identifier which many strings can map to, namely all that don't have a unique integer identifier).

    By default, a tokeniser of this class is also able to enumerate the types that belong to its identifiers.
    """
    # Implementation note:
    #     Tokenisers that are not injective (i.e. many tokens can map to the same ID) can still be modelled as a bijective
    #     tokeniser, but with an extra mapping between their output space and the vocabulary's input space. In that case, the
    #     vocabulary contains surrogate keys that have no surface-level meaning (e.g. V = {"token1": 1, "token2": 2, ...}
    #     and M = {"cat": "token1", "dog": "token1", "giraffe": "token1", "house": "token2", ...}) and the implementer can
    #     choose which strings to expose instead.

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials]):
        super().__init__(preprocessor=preprocessor)
        self.vocab = vocab
        self._accept_all_types = False  # if True, overrides hasType() by always returning True.

    # Iteration:

    def types(self) -> Iterable[str]:
        """The strings that can be produced as tokens by this tokeniser. Does not include specials because they aren't strings."""
        return self.vocab.keys()

    def ids(self) -> Iterable[int]:
        """The IDs belonging to the tokens produced by .types() (not necessarily in the right order)."""
        return self.vocab.values()

    # Conversion:

    def typeToId(self, t: str) -> int:
        """Convert the given type/token to an ID. Note: explicitly NOT meant for specials. Use .vocab.specials for this."""
        try:  # try-except is the fastest method to check+return a key in use cases where most lookups are valid (https://stackoverflow.com/a/28860508/9352077).
            return self.vocab[t]
        except:
            if self.vocab.UNK is not None:
                return self.vocab.UNK
            else:
                raise MissingUnkError(f"Token '{t}' has no ID, and the Vocab defines no UNK ID.")

    def idToType(self, i: int) -> str:
        try:
            return self.vocab.inverse[i]
        except:
            if i in self.vocab.specials:
                return ""
            else:
                raise RuntimeError(f"Unknown ID {i} cannot be decoded to a string type nor a special.")

    # Membership:

    def hasType(self, t: str) -> bool:
        return self._accept_all_types or t in self.vocab

    def hasId(self, i: int) -> bool:
        return i in self.vocab.inverse

    def enableInfiniteDomain(self, enable: bool):
        """
        If set to True, .types() and .ids() will still output the actual domain and range of the dictionary vocab,
        but .hasType() will always return True.

        This is technically incorrect, but has its uses. In particular, use this if you want to test the behaviour of a
        TokeniserWithVocabulary when its output isn't constrained.
        """
        self._accept_all_types = enable


def prepare_tokenise_decode(string: str, tokeniser: Tokeniser, preprocessor: Preprocessor) -> list[str]:
    """
    Tokenise, but afterwards, run each produced token back through the (inverse of) the pretokeniser + (invertible) mappings.

    Note: the preprocessor should be the effective preprocessor. E.g.: if you're using SentencePiece, this is your actual
    preprocessor followed by a mapper that adds the KudoSpaceMarker as a prefix.
    """
    return preprocessor.undo_per_token(tokeniser.prepareAndTokenise(string))
