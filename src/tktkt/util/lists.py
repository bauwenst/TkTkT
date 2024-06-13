from typing import Any, List, Iterable, Callable
from pathlib import Path


def fileToList(path: Path, include_empty_lines=True) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if include_empty_lines or line.strip()]


def intercalate(lst: list, new_element):
    """
    Insert a new element in between every existing element of a list.
    """
    new_list = []
    for old_element in lst:
        new_list.append(old_element)
        new_list.append(new_element)
    return new_list[:-1]


class NonePlaceholder:  # Replacement for None when None is actually a legitimate value.
    pass

def foldSpans(lst: list, only_delete_specific_value: Any=NonePlaceholder()):
    """
    Collapses contiguous spans of equal elements into just one element.
    Similar to the beta function in CTC loss.
    """
    if not lst:
        return []

    new_lst = [lst[0]]
    previous = lst[0]
    for i in range(1,len(lst)):
        current = lst[i]
        if current != previous:  # Different consecutive values are always kept.
            new_lst.append(current)
        elif not isinstance(only_delete_specific_value, NonePlaceholder) and current != only_delete_specific_value:  # The same consecutive values are also kept if they are not the specific element you want to delete.
            new_lst.append(current)

        previous = current

    return new_lst


def keepFirst(iterable: Iterable):
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


def mapExtend(f: Callable, iterable: Iterable) -> Iterable:
    """
    Same as map() except the per-element function produces iterables, which are yielded in turn as if they all belong
    to one long sequence.
    """
    for element in iterable:
        for piece in f(element):
            yield piece


def count(iterable: Iterable):
    total = 0
    for _ in iterable:
        total += 1
    return total
