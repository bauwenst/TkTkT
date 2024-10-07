from typing import TypeVar, Dict, List
from collections import OrderedDict

K = TypeVar("K")
V = TypeVar("V")


def invertdict(d: Dict[K,V], noninjective_ok=True) -> Dict[V,K]:
    """
    Keys become values, values become keys.
    Values must be hashable in that case.
    """
    d_inv = OrderedDict((v,k) for k,v in d.items())  # If the given dictionary is ordered, the resulting dictionary keeps that order.

    # Uh oh!
    if not noninjective_ok and len(d) != len(d_inv):
        values_with_multiple_keys = dict()

        # Redo the construction and keep track of keys with duplicate values.
        d_inv = dict()
        for k,v in d.items():
            if v in d_inv:  # Already in d_inv, so another key had this value.
                if v not in values_with_multiple_keys:
                    values_with_multiple_keys[v] = [d_inv[v]]
                values_with_multiple_keys[v].append(k)
            else:
                d_inv[v] = k

        raise ValueError(f"Dictionary could not be inverted because it wasn't injective. The following values were associated with more than one key: {values_with_multiple_keys}")

    return d_inv


def insertKeyAlias(d: Dict[K,V], existing_key: K, alias_key: K) -> Dict[K,V]:
    """
    In-place, but still returns the given dictionary.
    """
    assert existing_key in d
    d[alias_key] = d[existing_key]
    return d


def substituteKey(d: Dict[K,V], existing_key: K, new_key: K) -> Dict[K,V]:
    """
    In-place, but still returns the given dictionary.
    """
    assert existing_key in d and new_key not in d
    d[new_key] = d.pop(existing_key)
    return d


def getByValue(d: Dict[K,V], value: V) -> List[K]:
    return [k for k,v in d.items() if v == value]


def argmax(d: Dict[K,V]) -> List[K]:
    """Finds the keys belonging to the largest value in the dictionary."""
    return getByValue(d, max(d.values()))


def kargmax(d: Dict[K,V], k: int) -> List[List[K]]:
    """Finds the keys belonging to the k unique largest values in the dictionary. Could be O(k). I do it in O(n log(n))."""
    if k < 0:
        raise ValueError(f"k-argmax only exists for positive integers k. Received {k}.")

    values = set(d.values())
    if len(values) < k:
        raise ValueError(f"Could not get {k}-argmax: there are only {len(values)} unique values.")

    top_values = sorted(values, reverse=True)[:k]
    top_value_mapping = {v:i for i,v in enumerate(top_values)}

    key_buckets = [[] for _ in range(k)]
    for k,v in d.items():
        if v in top_value_mapping:
            key_buckets[top_value_mapping[v]].append(k)

    return key_buckets