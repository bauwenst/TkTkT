"""
General new types.
"""
from typing import Protocol, TypeVar, Iterable, Callable, Iterator, Union, Dict
from abc import abstractmethod
from datasets import Dataset, IterableDataset
from functools import partial

HuggingfaceDataset = Union[Dataset, IterableDataset]
Number = TypeVar("Number", bound=Union[int,float])

T = TypeVar("T")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

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
        from .iterables import streamProgress
        return self._iterable.__iter__() if not self._tqdm else streamProgress(self._iterable).__iter__()

    def tqdm(self) -> "NamedIterable[T]":
        self._tqdm = True
        return self

    def map(self, func: Callable[[T],T2]) -> "NamedIterable[T2]":
        """Refer to the documentation of the `mapped` class."""
        return NamedIterable(mapped(func, self), name=self.name)

    def flatmap(self, func: Callable[[T],Iterable[T2]]) -> "NamedIterable[T2]":
        """Refer to the documentation of `flatmapped` class."""
        return NamedIterable(flatmapped(func, self), name=self.name)

    def wrap(self, func: Callable[[Iterable[T]], Iterable[T2]]) -> "NamedIterable[T2]":
        """Refer to the documentation of `wrappediterable` class."""
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
        from .iterables import mapExtend
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


class anypartial(partial):
    """
    An improved version of functools.partial which accepts ellipsis (...) as a placeholder, so that not just the
    leftmost positional arguments can be captured, but any positional argument. For example:
        def f(x,y,z,w):
            return x + y*z/w

        g = anypartial(f, 10, ..., ..., 2)
        assert g(6,7) == 10 + 6*7/2 == 31  # Rather than 10 + 2*6/7.

    Taken from https://stackoverflow.com/a/66274908/9352077.
    """
    def __call__(self, *args, **keywords):  # We store self.func, self.args, self.keywords.
        keywords = {**self.keywords, **keywords}  # Keywords args have no order.
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)  # We run through the stored args. If an ellipsis is found, we advance the given args by one and replace the ellipsis by what was skipped.
        return self.func(*args, *iargs, **keywords)  # We now replay the stored args (with filled-in ellipses), then the advanced given args, and then keywords.


class dictget:
    # For some strange reason, just subclassing anypartial is not allowed.
    # def __init__(self, key: T):
    #     super().__init__(dict.get, ..., key)

    def __init__(self, key: T):
        self._partial = anypartial(dict.get, ..., key)

    def __call__(self, d: Dict[T,T2]) -> T2:
        return self._partial(d)


class pipe(Callable[[T],T3]):

    def __init__(self, f1: Callable[[T], T2], f2: Callable[[T2], T3]):
        self.f1 = f1
        self.f2 = f2

    def __call__(self, *args, **kwargs):
        return self.f2(self.f1(*args, **kwargs))


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
