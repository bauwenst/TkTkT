"""
Operations that can stream an input and output a stream in return.

Note that all iterators are iterable (have __iter__), but not all iterables are iterators (have __next__). That is:
iterators are a special kind of iterable. More specific than just any iterable.
Also -- but this is not part of Python's typing -- iterators are consumable only once.

The outputs of all the functions below are explicitly iterables with __next__ method that are consumable only once,
hence why the output is marked as an Iterator.
"""
from typing import Any, List, Iterable, Callable, Generator, TypeVar, Union, Optional, Iterator
from pathlib import Path
from tqdm.auto import tqdm
import numpy.random as npr

T = TypeVar("T")
T2 = TypeVar("T2")
Number = TypeVar("Number", bound=Union[int,float])

def streamLines(path: Path, include_empty_lines=True) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return filter(lambda line: include_empty_lines or len(line),
                      map(lambda line: line.strip(),
                          handle))


class _NonePlaceholder:  # Replacement for None when None is actually a legitimate value.
    pass
_NONE = _NonePlaceholder()


def intercalate(lst: Iterable[T], new_element: T2) -> Iterator[Union[T,T2]]:
    """
    Insert a new element in between every existing element of an iterable.
    """
    buffer = _NONE
    for e in lst:
        if buffer != _NONE:  # Will not trigger on the first iteration, in order to fill the buffer.
            yield buffer
            yield new_element
        buffer = e  # In the last iteration, the loop will exit while the last element has not been yielded yet.
    yield buffer


def foldSpans(lst: Iterable[T], only_delete_specific_value: Any=_NONE) -> Iterator[T]:
    """
    Collapses contiguous spans of equal elements into just one element.
    Similar to the beta function in CTC loss.
    """
    previous = _NONE
    for thing in lst:
        if previous == _NONE or thing != previous or (only_delete_specific_value != _NONE and thing != only_delete_specific_value):  # Different consecutive values are always kept. The same consecutive values are also kept if they are not the specific element you want to delete.
            yield thing
        previous = thing


def keepFirst(iterable: Iterable[T], key: Callable[[T],T2]=None) -> Iterator[T]:
    """
    Only keeps the first instance of each unique value in the list.
    Currently requires the elements to be hashable to keep the function O(N). Can be made O(N²) by just using == instead.
    """
    if key is None:
        key = lambda x: x

    seen_keys = set()
    for e in iterable:
        k = key(e)
        if k in seen_keys:
            continue
        else:
            seen_keys.add(k)
            yield e


def mapExtend(f: Callable[[T],Iterable[T2]], iterable: Iterable[T]) -> Iterator[T2]:
    """
    Same as map() except the per-element function produces iterables, which are yielded in turn as if they all belong
    to one long sequence.
    """
    for element in iterable:
        for piece in f(element):
            yield piece


def cat(iterable: Iterable[Iterable[T]]) -> Iterator[T]:
    return mapExtend(lambda x: x, iterable)


IterableOrT = Union[T, Iterable['IterableOrT[T]']]
def flattenRecursively(nested_iterable: IterableOrT[T]) -> Iterator[T]:
    try:
        for thing in nested_iterable:
            yield from flattenRecursively(thing)
    except:
        yield nested_iterable


def drop(n: int, iterable: Iterable[T]) -> Generator[T, None, None]:
    for i, thing in enumerate(iterable):
        if i < n:
            continue
        yield thing


def take(n: int, iterable: Iterable[T]) -> Generator[T, None, None]:
    if n > 0:
        for i, thing in enumerate(iterable):
            yield thing
            if i+1 == n:
                break


def takeAfterShuffle(n: int, known_size: int, iterable: Iterable[T], rng=npr.default_rng(seed=0)) -> Generator[T, None, None]:
    if n > known_size:
        raise ValueError(f"Cannot take {n} items from an iterable whose size is reported as being only {known_size}.")

    indices = set(rng.integers(low=0, high=known_size, size=n))
    for i,thing in enumerate(iterable):
        if i in indices:
            indices.pop(i)
            yield thing


def takeRandomly(p: float, iterable: Iterable[T], rng=npr.default_rng(seed=0)) -> Generator[T, None, None]:
    for thing in iterable:
        if rng.random() < p:
            yield thing


def filterOptionals(iterable: Iterable[Optional[T]]) -> Iterator[T]:
    for thing in iterable:
        if thing is not None:
            yield thing


def cumsum(iterable: Iterable[Number]) -> Iterator[Number]:
    total = 0
    for n in iterable:
        total += n
        yield total


def streamPrint(iterable: Iterable[T]) -> Iterator[T]:
    for thing in iterable:
        print(thing)
        yield thing


def streamProgress(iterable: Iterable[T], show_as: Optional[str]=None, known_size: Optional[int]=None) -> Iterable[T]:
    return tqdm(iterable, desc=show_as, total=known_size, smoothing=0.1)


# Endpoints below

def count(iterable: Iterable[T]) -> int:
    total = 0
    for _ in iterable:
        total += 1
    return total


def allEqual(iterable: Iterable[T]) -> bool:
    value = _NONE
    for thing in iterable:
        if value is _NONE:
            value = thing
            continue

        if thing != value:
            return False

    return True

def transpose(matrix: Iterable[Iterable[T]]) -> List[List[T]]:
    new_matrix = []
    for row in matrix:
        for y,e in enumerate(row):
            if y >= len(new_matrix):
                new_matrix.append([])
            new_matrix[y].append(e)

    return new_matrix
