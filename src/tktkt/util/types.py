"""
General new types.
"""
from abc import abstractmethod
from typing import Protocol, TypeVar, Iterable, Callable, Iterator, Union
from datasets import Dataset, IterableDataset

HuggingfaceDataset = Union[Dataset, IterableDataset]

T = TypeVar("T")
T2 = TypeVar("T2")

class NamedIterable(Iterable[T]):  # This T is so that type signatures like NamedIterable[str] actually cause type inference for return values once iterating.
    """
    An iterable that has a string attached to it. Handy when e.g. you have a streamable corpus and want to name the
    results based on the name of the corpus.
    """
    def __init__(self, iterable: Iterable[T], name: str):  # This T is so that the above T is be inferred from the constructor if there is no type signature.
        self._iterable = iterable
        self.name = name

        if hasattr(iterable, "__next__"):
            raise TypeError("The given iteraBLE is an iteraTOR, and hence may not be re-iterable.")

    def __iter__(self):
        return self._iterable.__iter__()

    def map(self, func: Callable[[T],T2]) -> "NamedIterable[T2]":
        return NamedIterable(mapped(func, self), name=self.name)


class mapped(Iterable[T2]):
    """
    Reusable version of map(). The latter is consumed after iterating over it once, even if the mapped iterable isn't.
    """
    def __init__(self, func: Callable[[T],T2], iterable: Iterable[T]):
        self._function = func
        self._iterable = iterable

    def __iter__(self) -> Iterator[T2]:
        return map(self._function, self._iterable)


class Comparable(Protocol):
    """
    Anything that has "strictly less than" defined.
    https://stackoverflow.com/a/65224102/9352077

    You could technically also `from _typeshed import SupportsLessThan` but that uses Any as argument types rather than
    just requiring being comparable to an object of the same type.
    """

    @abstractmethod
    def __lt__(self: "CT", other: "CT") -> bool:
        pass

CT = TypeVar("CT", bound=Comparable)  # Defined so it can be used in the signature above.
