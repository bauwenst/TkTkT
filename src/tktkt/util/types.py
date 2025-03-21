"""
General new types.
"""
from abc import abstractmethod
from typing import Protocol, TypeVar, Iterable, Callable, Iterator, Union
from datasets import Dataset, IterableDataset

from .iterables import mapExtend, streamProgress

HuggingfaceDataset = Union[Dataset, IterableDataset]

T = TypeVar("T")
T2 = TypeVar("T2")

class NamedIterable(Iterable[T]):  # This T is so that type signatures like NamedIterable[str] actually cause type inference for return values once iterating.
    """
    An iterable that has a string attached to it. Handy when e.g. you have a streamable corpus and want to name the
    results based on the name of the corpus.
    """
    def __init__(self, iterable: Iterable[T], name: str):  # This T is so that the above T is be inferred from the constructor if there is no type signature.
        self.name = name
        self._iterable = iterable
        self._tqdm = False

        if hasattr(iterable, "__next__"):
            raise TypeError("The given iteraBLE is an iteraTOR, and hence may not be re-iterable.")

    def __iter__(self):
        return self._iterable.__iter__() if not self._tqdm else streamProgress(self._iterable).__iter__()

    def tqdm(self) -> "NamedIterable[T]":
        self._tqdm = True
        return self

    def map(self, func: Callable[[T],T2]) -> "NamedIterable[T2]":
        return NamedIterable(mapped(func, self), name=self.name)

    def flatmap(self, func: Callable[[T],Iterable[T2]]) -> "NamedIterable[T2]":
        return NamedIterable(flatmapped(func, self), name=self.name)

    def wrap(self, func: Callable[[Iterable[T]], Iterable[T2]]) -> "NamedIterable[T2]":
        return NamedIterable(wrappediterable(func, self), name=self.name)


class mapped(Iterable[T2]):
    """
    Reusable version of map(). The latter is consumed after iterating over it once, even if the mapped iterable isn't.
    Applies the given function to every element in the iterable.
    """
    def __init__(self, func: Callable[[T],T2], iterable: Iterable[T]):
        self._function = func
        self._iterable = iterable

    def __iter__(self) -> Iterator[T2]:
        return map(self._function, self._iterable)


class flatmapped(Iterable[T2]):
    """
    Like mapped() except for functions that produce iterables that need to be concatenated.
    """
    def __init__(self, func: Callable[[T],Iterable[T2]], iterable: Iterable[T]):
        self._function = func
        self._iterable = iterable

    def __iter__(self) -> Iterator[T2]:
        return mapExtend(self._function, self._iterable)


class wrappediterable(Iterable[T2]):
    """
    Applies the given function once, to the iterable's iterator itself, NOT to its elements nor to the iterable.
    This is really a generalisation of map(), which applies the following function to the iterable:

    def func(item_function, iterator):
        for thing in iterator:
            yield item_function(thing)
    """
    def __init__(self, func: Callable[[Iterable[T]], Iterable[T2]], iterable: Iterable[T]):
        self._function = func
        self._iterable = iterable

    def __iter__(self):
        return self._function(self._iterable.__iter__())


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
