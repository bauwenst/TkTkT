"""
Operations that can stream an input and output a stream in return.
"""
from typing import Any, List, Iterable, Callable, Generator, TypeVar, Union, Optional
from pathlib import Path
from tqdm.auto import tqdm

T = TypeVar("T")
T2 = TypeVar("T2")


def streamLines(path: Path, include_empty_lines=True) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return filter(lambda line: include_empty_lines or len(line),
                      map(lambda line: line.strip(),
                          handle))


class _NonePlaceholder:  # Replacement for None when None is actually a legitimate value.
    pass
_NONE = _NonePlaceholder()


def intercalate(lst: Iterable[T], new_element: T2) -> Iterable[Union[T,T2]]:
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


def foldSpans(lst: Iterable[T], only_delete_specific_value: Any=_NONE) -> Iterable[T]:
    """
    Collapses contiguous spans of equal elements into just one element.
    Similar to the beta function in CTC loss.
    """
    previous = _NONE
    for thing in lst:
        if previous == _NONE or thing != previous or (only_delete_specific_value != _NONE and thing != only_delete_specific_value):  # Different consecutive values are always kept. The same consecutive values are also kept if they are not the specific element you want to delete.
            yield thing
        previous = thing


def keepFirst(iterable: Iterable[T]) -> Iterable[T]:
    """
    Only keeps the first instance of each unique value in the list.
    Currently requires the elements to be hashable to keep the function O(N). Can be made O(N²) by just using == instead.
    """
    values = set()
    for e in iterable:
        if e in values:
            continue
        else:
            values.add(e)
            yield e


def mapExtend(f: Callable[[T],Iterable[T2]], iterable: Iterable[T]) -> Iterable[T2]:
    """
    Same as map() except the per-element function produces iterables, which are yielded in turn as if they all belong
    to one long sequence.
    """
    for element in iterable:
        for piece in f(element):
            yield piece


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


def filterOptionals(iterable: Iterable[Optional[T]]) -> Iterable[T]:
    for thing in iterable:
        if thing is not None:
            yield thing


def streamPrint(iterable: Iterable[T]) -> Iterable[T]:
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


def transpose(matrix: Iterable[Iterable[T]]) -> List[List[T]]:
    new_matrix = []
    for row in matrix:
        for y,e in enumerate(row):
            if y >= len(new_matrix):
                new_matrix.append([])
            new_matrix[y].append(e)

    return new_matrix
