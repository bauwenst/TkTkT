from typing import List, Optional, Iterator, Set
from abc import abstractmethod, ABC

from ..preparation.boundaries import BoundaryMarker


class _PreprocessorComponent(ABC):
    """
    Defines properties of the preprocessor that can be registered by any one of its contents.
    """
    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        return None

    def getAlphabet(self) -> Optional["FiniteCharacterSet"]:
        return None

    @abstractmethod
    def __iter__(self) -> Iterator["_PreprocessorComponent"]:
        pass


class _PreprocessorComponentSequence(_PreprocessorComponent):
    """
    Implementations of the preprocessor properties for components that don't yield themselves when being iterated over,
    which is the case for sequence classes.
    """

    def getAlphabet(self) -> Optional["FiniteCharacterSet"]:  # FIXME: Known issue: the marker may not be in this alphabet.
        alphabet = None
        for component in self:  # TODO: Interestingly, because PretokeniserSequence and TextMapperSequence don't yield themselves, the recursive call to getAlphabet happens only in leaves, and the traversing is done by this for statement. But is this good design?
            alphabet = component.getAlphabet() or alphabet
        return alphabet

    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        marker = None
        for component in self:
            marker = component.getBoundaryMarker() or marker
        return marker


class TextMapper(_PreprocessorComponent):
    """
    Turns one string into another string. By default, this is not invertible.
    """
    @abstractmethod
    def convert(self, text: str) -> str:
        pass

    def __iter__(self) -> Iterator[_PreprocessorComponent]:
        yield self


class InvertibleTextMapper(TextMapper):
    """
    Turns one string into another string and can deduce the original from the result.
    """
    @abstractmethod
    def invert(self, text: str) -> str:
        pass


class FiniteCharacterSet(TextMapper):
    """
    Defines a set of characters that is used to encode any Unicode string.
    The most popular such set is are the characters used by HuggingFace to represent UTF-8 bytes, which is also lossless.
    """
    @abstractmethod
    def getCharacters(self) -> List[str]:
        pass

    def getAlphabet(self) -> Optional["FiniteCharacterSet"]:
        return self


class Pretokeniser(_PreprocessorComponent):
    """
    A note on the design of pretokenisers: you want pretokenisers to be invertible to get back to a single string, but
    especially with sequentially applied pretokenisers, this is ill-defined (because fundamentally, list.extend isn't
    invertible). There are three ways you can implement un-splitting:

            1. "".join(map(invertOne, tokens)) or
            2. "".join(invertMany(tokens)) or
            3. invertString("".join(tokens))

    This matters because for example, the ALBERT ULM decoder is something like "".join(tokens).replace("_", " ").strip(),
    which means that inverting single tokens can never produce spaces. Spaces are only protected when they appear
    in tokens that are surrounded by other tokens during decoding. This is an example of an invertMany, like all HuggingFace pretokenisers.

    1 and 3 are special cases of 2, where respectively
        invertMany(tokens) == map(invertOne, tokens)
        invertMany(tokens) == [invertString("".join(tokens))]
    A class may elect to use such implementations.
    """

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """
        Split an entire text (e.g. a sentence) into smaller pre-token strings.
        It is these pre-token strings that will SEPARATELY be passed to the tokeniser.
        """
        pass

    @abstractmethod
    def invertTokens(self, pretokens: List[str]) -> List[str]:
        """
        Invert any string transformations applied in the process of splitting a string into pretokens.
        May also apply a merging operation into a smaller list if that is appropriate.
        """
        pass  # For example, [invertToken(t) for t in tokens], but you should make this explicit because it's not necessarily the case.

    def invertToken(self, pretoken: str) -> str:
        """
        Takes (part of) a pretoken and undoes any character transformations that were applied during pretokenisation.
        Tricky, because this isn't necessarily possible: for example, if you mapped from 1 Unicode charactere to >1 bytes,
        and the tokeniser separated those bytes into separate tokens, converting one token at a time will not work.
        """
        return "".join(self.invertTokens([pretoken]))

    def unsplit(self, tokens: List[str]) -> str:
        """
        Inverts the splitting operation.
        """
        return "".join(self.invertTokens(tokens))

    def __call__(self):  # Just in case you accidentally add parentheses to an already-instantiated Pretokeniser object.
        return self

    def __iter__(self) -> Iterator[_PreprocessorComponent]:
        yield self

    def getName(self):
        return self.__class__.__name__


class Preprocessor(_PreprocessorComponentSequence):
    """
    Applies the following transformations to a string of text:

        raw text
          v  one-time mapping
        clean text                       clean text
          v  invertible mapping               ^
        altered text                     altered text
          v  splitter                         ^
        pretokens   --- (tokeniser) ---->   tokens
    """

    def __init__(self, uninvertible_mapping: TextMapper=None, invertible_mapping: InvertibleTextMapper=None, splitter: Pretokeniser=None):
        from ..preparation.mappers import IdentityMapper  # Import here to prevent circular import when importing from .mappers or .splitters.
        from ..preparation.splitters import IdentityPretokeniser

        self.irreversible: TextMapper         = uninvertible_mapping or IdentityMapper()
        self.reversible: InvertibleTextMapper = invertible_mapping   or IdentityMapper()
        self.splitter: Pretokeniser           = splitter             or IdentityPretokeniser()

    def __call__(self):  # Just in case you accidentally add parentheses to an already-instantiated Preprocessor object.
        return self

    def do(self, text: str) -> List[str]:
        return self.splitter.split(self.reversible.convert(self.irreversible.convert(text)))

    def undo(self, tokens: List[str]) -> str:
        return self.reversible.invert(self.splitter.unsplit(tokens))

    def undo_per_token(self, tokens: List[str]) -> List[str]:
        return [self.reversible.invert(self.splitter.invertToken(token)) for token in tokens]

    def __iter__(self) -> Iterator[_PreprocessorComponent]:
        yield from self.irreversible
        yield from self.reversible
        yield from self.splitter
