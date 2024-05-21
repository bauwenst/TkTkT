from typing import Any


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

def reduceSpans(lst: list, only_delete_specific_value: Any=NonePlaceholder()):
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
