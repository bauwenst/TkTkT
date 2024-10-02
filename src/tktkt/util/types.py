"""
General new types.
"""
from abc import abstractmethod
from typing import Protocol, TypeVar, Iterable, Callable

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

    def __iter__(self):
        return self._iterable.__iter__()

    def map(self, f: Callable[[T],T2]) -> "NamedIterable[T2]":
        return NamedIterable(map(f, self), name=self.name)


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
