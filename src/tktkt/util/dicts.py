from typing import TypeVar, Dict, List, Generic, Callable
from collections import OrderedDict, Counter

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
    """Finds the keys belonging to the largest value in the dictionary. Could be ~n. I do it in ~2n."""
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


class ChainedCounter(Counter[K], Generic[K]):
    """
    Similar to a Counter, except it buckets N observations at a time into sub-counters internally. That way, it is
    possible to compute averages across fixed-size sample windows, as is done e.g. for MATTR (see Covington & McFall, 2010).

    Subclass of Counter, which means it also inherits the data structure used by Counter to store and retrieve items
    globally. The sub-counters are hence purely for illustrative purposes; most methods are still done in O(1) time by
    referring to that global data structure of super().
    """

    def __init__(self, max_subcounter_size: int):
        super().__init__()  # Because this is a subclass of Counter, the super class has a "master data structure".
        self._counters = [Counter()]
        self._current_size = 0
        self._max_size     = max_subcounter_size

    def averageOverCounters(self, counter_function: Callable[[Counter],float]) -> float:
        return 1/len(self._counters) * sum(counter_function(counter) for counter in self._counters)

    def __setitem__(self, key, value):
        """
        Assume that this is never coming from an expression "object[key] = value" and always from some kind of increment
        that already includes the current value. (Note: this means that if you try object[key] = value, what will happen
        is that it will add  value - object[key]  to the subcounters, not value.)
        """
        # Update the master counter
        increment = value - self[key]  # Before the master counter is updated, get the difference with the old value.
        super().__setitem__(key, value)

        assert increment >= 0, "Cannot take values out of a chained counter."

        # Update the sub-counters
        while increment > 0:
            new_cur = min(self._max_size, self._current_size + increment)
            delta_in_current_counter = new_cur - self._current_size
            self._counters[-1][key] += delta_in_current_counter

            increment -= delta_in_current_counter
            self._current_size = new_cur
            if self._current_size == self._max_size:
                self._current_size = 0
                self._counters.append(Counter())

    def get(self, key, default=None):
        # For some unknown reason, Counter's implementation of .get() does not use .__getitem__(). As it turns out, the
        # .update() method uses .get() rather than counter[key], and hence to support .update(), it doesn't suffice to override .__getitem__() only.
        if key in self:
            return self[key]
        else:
            return default
