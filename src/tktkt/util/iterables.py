"""
Operations that can stream an input and output a stream in return.

Note that all iterators are iterable (have __iter__), but not all iterables are iterators (have __next__). That is:
iterators are a special kind of iterable. More specific than just any iterable.
Also -- but this is not part of Python's typing -- iterators are consumable only once.

The outputs of all the functions below are explicitly iterables with __next__ method that are consumable only once,
hence why the output is marked as an Iterator.
"""
from typing import Any, List, Iterable, Callable, Generator, TypeVar, Union, Optional, Iterator, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import numpy.random as npr

from .types import Number, T, T2, CT
from .printing import warn


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


def take(n: int, iterable: Iterable[T], exact: bool=False) -> Generator[T, None, None]:
    if n > 0:
        i = 0
        for thing in iterable:
            i += 1
            yield thing
            if i == n:
                break

        if i < n:
            if exact:
                raise IndexError(f"Requested exactly {n} items from iterable, but it ran out after {i}.")
            else:
                warn(f"Requested {n} items from iterator, but it ran out after {i}.")


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


def zipSelf(iterable: Iterable[T], offset: int=1) -> Iterator[Tuple[T,T]]:
    """
    Zip an iterable with an offset copy of itself.
    The lagged copy will be first. For example,
        A B C D E F G
    with offset 2 will be zipped into
        (A,C) (B,D) (C,E) (D,F) (E,G)
    """
    assert offset > 0

    buffer = []
    for thing in iterable:
        buffer.append(thing)
        if len(buffer) < offset+1:
            continue

        yield buffer.pop(0), thing


def streamPrint(iterable: Iterable[T]) -> Iterator[T]:
    for thing in iterable:
        print(thing)
        yield thing


def streamProgress(iterable: Iterable[T], show_as: Optional[str]=None, known_size: Optional[int]=None) -> Iterable[T]:
    return tqdm(iterable, desc=show_as, total=known_size, smoothing=0.1)


#################
### Endpoints ###
#################

def at(index: int, iterable: Iterable[T]) -> T:
    for i, thing in enumerate(iterable):
        if i == index:
            return thing
    else:
        raise IndexError(f"Index {index} out of range of iterable.")


def first(iterable: Iterable[T]) -> T:
    for thing in iterable:
        return thing  # Immediately exits the loop and the function.
    raise IndexError("There is no first item in an empty iterable.")


def last(iterable: Iterable[T]) -> T:
    last_thing = _NONE
    for thing in iterable:
        last_thing = thing
    if isinstance(last_thing, _NonePlaceholder):
        raise IndexError("There is no last item in an empty iterable.")
    return last_thing


def fst(t: Tuple[T,T2]) -> T:
    return t[0]


def snd(t: Tuple[T,T2]) -> T2:
    return t[1]


def indexSpan(subiterable: Iterable[T], iterable: Iterable[T]) -> Optional[Tuple[int,int]]:
    """
    Find the first occurrence of the given subiterable in the other iterable. Also returns the exclusive end index.

    Note that this is equivalent to string search. This implementation uses backtracking and is, in the worst case,
    O(N*M), whereas algorithms like KMP are faster.
    """
    target = list(subiterable)
    buffer = []
    if not target:
        return (0,0)  # Trivial empty span.

    for last_idx,thing in enumerate(iterable):
        buffer.append(thing)
        if len(buffer) == len(target):
            if buffer == target:  # I assume this uses early stopping.
                return (last_idx+1-len(target), last_idx+1)
            buffer.pop(0)
    return None


def count(iterable: Iterable[T]) -> int:
    """Count the elements in the given iterable."""
    total = 0
    for _ in iterable:
        total += 1
    return total


def maxargmax(iterable: Iterable[CT]) -> Tuple[CT, int]:
    max_element    = None
    argmax_element = None
    for index, thing in enumerate(iterable):
        if max_element is None or max_element < thing:
            max_element    = thing
            argmax_element = index

    if max_element is None:
        raise IndexError("There were no items in the given iterable.")

    return max_element, argmax_element


def minargmin(iterable: Iterable[CT]) -> Tuple[CT, int]:
    min_element    = None
    argmin_element = None
    for index, thing in enumerate(iterable):
        if min_element is None or thing < min_element:
            min_element    = thing
            argmin_element = index

    if min_element is None:
        raise IndexError("There were no items in the given iterable.")

    return min_element, argmin_element


def allEqual(iterable: Iterable[T]) -> bool:
    value = _NONE
    for thing in iterable:
        if value is _NONE:
            value = thing
            continue

        if thing != value:
            return False

    return True


def areEquidistant(iterable: Iterable[Number], distance: Number) -> bool:
    return all(a+distance == b for a,b in zipSelf(iterable, offset=1))


def areContiguous(iterable: Iterable[int]) -> bool:
    return areEquidistant(iterable, distance=1)


def transpose(matrix: Iterable[Iterable[T]]) -> List[List[T]]:
    new_matrix = []
    for row in matrix:
        for y,e in enumerate(row):
            if y >= len(new_matrix):
                new_matrix.append([])
            new_matrix[y].append(e)

    return new_matrix
