from typing import TypeVar, Dict, List, Generic, Callable, Union, Iterator, Tuple
from typing_extensions import Self
from collections import OrderedDict, Counter
from pathlib import Path

import json
import dacite
import dataclasses
import numpy as np
import numpy.random as npr

from .timing import datetimeDashed
from .printing import inequality, warn
from .types import Number

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

        raise ValueError(f"Dictionary could not be inverted because it wasn't injective. The following values were associated with more than one key:\n{values_with_multiple_keys}")

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


def normaliseCounter(counts: Union[Counter[K], Dict[K,Union[int,float]]]) -> Dict[K,float]:
    total = sum(counts.values())
    return {t: c/total for t,c in counts.items()}


def dictToJson(data: dict, path_to_store: Path, do_indent: bool=True) -> Path:
    # Imputations
    if path_to_store.is_dir():
        path_to_store = path_to_store / datetimeDashed()
    path_to_store = path_to_store.with_suffix(".json")

    # Store
    with open(path_to_store, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4 if do_indent else None, ensure_ascii=False)
    return path_to_store

def dataclassToJson(dataclass_instance, path_to_store: Path, do_indent: bool=True) -> Path:
    return dictToJson(dataclasses.asdict(dataclass_instance), path_to_store, do_indent)


def jsonToDict(path_to_load: Path) -> dict:
    with open(path_to_load.with_suffix(".json"), "r", encoding="utf-8") as file:
        return json.load(file)


def jsonToDataclass(dataclass_type: type[V], path_to_load: Path) -> V:
    return dacite.from_dict(dataclass_type, jsonToDict(path_to_load))


def optionalDataclassToDict(dataclass_or_dict) -> dict:
    if dataclasses.is_dataclass(dataclass_or_dict):
        return dataclasses.asdict(dataclass_or_dict)
    elif isinstance(dataclass_or_dict, dict):
        return dataclass_or_dict
    else:
        raise TypeError(f"Unsupported type: {type(dataclass_or_dict)}")

K = TypeVar("K")
V = TypeVar("V")

class ChainedCounter(Counter[K], Generic[K]):
    """
    Similar to a Counter, except it buckets N observations at a time into sub-counters internally. That way, it is
    possible to compute averages across fixed-size sample windows, as is done e.g. for MATTR (see Covington & McFall, 2010).

    Subclass of Counter, which means it also inherits the data structure used by Counter to store and retrieve items
    globally. The sub-counters are hence purely for illustrative purposes; most methods are still done in O(1) time by
    referring to that global data structure of super().

    Note: for some reason, deepcopy() produces a nonsensical state for this class. Use the built-in copy method.
    """

    def __init__(self, max_subcounter_size: int, seed: int=0):
        super().__init__()  # Because this is a subclass of Counter, the super class has a "master data structure".
        self._counters: List[Counter[K]] = []
        self._size_of_last_counter = max_subcounter_size

        self._max_size = max_subcounter_size
        self._seed = seed
        self._rng = npr.default_rng(seed)

    def subcounterAmount(self) -> int:
        return len(self._counters)

    def subcounterSizes(self) -> List[int]:
        return list(self.mapCounters(lambda c: c.total()))

    def mapCounters(self, counter_function: Callable[[Counter],V]) -> Iterator[V]:
        return map(counter_function, self._counters)  # If the function expects two arguments, this will crash.

    def averageOverCounters(self, counter_function: Callable[[Counter],Number]) -> Number:
        weights = self.subcounterSizes()
        values  = list(self.mapCounters(counter_function))
        return sum(w*v for w,v in zip(weights,values))/sum(weights) if weights else 0

    def averageOverCountersAndIndices(self, counter_function: Callable[[int,Counter],Number]) -> Number:
        """Use this if you need a list in your counter function indexed based on the counter."""
        weights = self.subcounterSizes()
        values = list(map(counter_function, range(len(self._counters)), self._counters))
        return sum(w*v for w,v in zip(weights,values))/sum(weights) if weights else 0

    def serialise(self) -> Tuple[List[Dict[K,int]], int]:
        return [dict(c.items()) for c in self._counters], self._max_size

    @classmethod
    def deserialise(cls, serialised: List[Dict[K,int]], max_size: int, seed: int) -> "ChainedCounter[K]":
        supercounter = ChainedCounter(max_size, seed)
        for subcounter in serialised:
            for k,v in subcounter.items():
                supercounter[k] += v
        return supercounter

    ####################################################################################################################

    def __setitem__(self, key: K, value):
        """
        Assume that this is never coming from an expression "object[key] = value" and always from some kind of increment
        that already includes the current value. (Note: this means that if you try object[key] = value, what will happen
        is that it will add  `value - object[key]`  to the subcounters, not `value`.)
        """
        # Update the master counter
        increment = value - self[key]  # Before the master counter is updated, get the difference with the old value.
        super().__setitem__(key, value)

        assert increment >= 0, "Cannot take values out of a chained counter."

        # Update the sub-counters
        while increment > 0:
            if self._size_of_last_counter == self._max_size:  # Putting this at the end of the loop would cause an empty last counter to exist sometimes.
                self._size_of_last_counter = 0
                self._counters.append(Counter())

            new_cur = min(self._max_size, self._size_of_last_counter + increment)
            delta_in_current_counter = new_cur - self._size_of_last_counter
            self._counters[-1][key] += delta_in_current_counter

            increment -= delta_in_current_counter
            self._size_of_last_counter = new_cur

    def pop(self, key: K) -> int:
        """
        Reverse of all __setitem__ calls to the same key.
        Unlike subtraction, this is actually feasible, since you know which counters to remove the key from (all of them).
        """
        for subcounter in self._counters:
            if key in subcounter:
                subcounter.pop(key)
        self._garbageCollect()
        return super().pop(key)

    def __add__(self, other: "ChainedCounter[K]") -> "ChainedCounter[K]":
        """
        Adding two counters together.

        Since it's impossible to know in what order the samples of the given counter came in within its subcounters,
        we assume they came in randomly according to the distribution they appear to have.
        """
        if not isinstance(other, ChainedCounter):
            raise NotImplementedError
        if self._max_size != other._max_size:
            raise NotImplementedError("See the FIXME note below.")

        # # Short implementation (but with linear complexity a.f.o. corpus size, so very slow):
        # sum_counter = ChainedCounter(self._max_size, seed=self._rng.integers(0, 1_000_000))
        # for counter in self._counters:
        #     for k,v in counter.items():
        #         sum_counter[k] += v
        # for element in other._regenerate():
        #     sum_counter[element] += 1
        # assert sum_counter.total() == self.total() + other.total(), f"Expected {self.total()} + {other.total()} = {inequality(self.total() + other.total(), sum_counter.total())} found."
        # return sum_counter

        def update(counter_a: Counter, counter_b: Counter):
            for element,count in counter_b.items():
                counter_a[element] += count

        # Early exits
        sum_counter = self.copy()
        if not self._counters and self._max_size == other._max_size:
            return other.copy()
        elif not other._counters:
            return sum_counter

        # Fill the last subcounter of counter A with subcounters from counter B, starting at the end so counter B has minimal defragmentation to do.
        sum_counter.defragment()
        deficit = sum_counter._max_size - sum_counter._size_of_last_counter

        # - First test for an early exit, where the subcounters in B all get pooled into one subcounter of A.
        if deficit >= other.total():
            update(sum_counter, other)
            assert sum_counter.total() == self.total() + other.total(), f"Expected {self.total()} + {other.total()} = {inequality(self.total() + other.total(), sum_counter.total())} found."
            return sum_counter

        # - Now we know that the deficit of A's last subcounter can be filled without taking everything out of B.
        final_counter_idx     = other.subcounterAmount() - 1
        final_counter_samples = Counter()
        while deficit > 0:
            counter   = other._counters[final_counter_idx]
            available = counter.total()
            if deficit >= available:  # Easy, just add the entire subcounter.
                update(sum_counter, counter)
                final_counter_idx -= 1
                assert final_counter_idx >= 0
            else:  # Sample this final subcounter.
                final_counter_samples = other.sampleSubcounter(final_counter_idx, deficit)
                update(sum_counter, final_counter_samples)
                assert final_counter_samples.total() == deficit
            deficit -= available

        # Take the rest out of counter B, now from left to right. FIXME: When A has smaller max_size than B, this approach is incorrect. You'll take e.g. only half of the unique elements in one subcounter of B and fill one of A's subcounters with it. You should actually use sampleSubcounter() for this.
        for i in range(final_counter_idx):
            update(sum_counter, other._counters[i])
        update(sum_counter, other._counters[final_counter_idx] - final_counter_samples)  # For the last non-empty counter, don't include the samples you already included.

        assert sum_counter.total() == self.total() + other.total(), f"Expected {self.total()} + {other.total()} = {inequality(self.total() + other.total(), sum_counter.total())} found."
        return sum_counter

    def defragment(self, sort_subcounters_first: bool=False):
        """
        Re-pack values inside this counter so that any gaps left by popping from the subcounters are filled up again.
        """
        if not self._counters:
            return

        if sort_subcounters_first:
            self._counters.sort(key=lambda c: -c.total())  # Biggest subcounter first.
            self._garbageCollect()

        total_before_defragmenting = self.total()
        i = 0
        while i < self.subcounterAmount():
            deficit = self._max_size - self._counters[i].total()
            assert deficit >= 0  # Something is very wrong if this is false.

            j = i + 1  # Steal items from this counter.
            while deficit and j < self.subcounterAmount():
                next_counter = self._counters[j]
                available = next_counter.total()
                if deficit >= available:  # The next counter can't fill the gap, or just barely can. Copy in its entirety. No need to sample.
                    self._counters[i] += next_counter
                    self._counters[j] -= next_counter
                    deficit -= available
                    j += 1
                else:
                    counts_to_commit = self.sampleSubcounter(j, deficit)
                    self._counters[i] += counts_to_commit
                    self._counters[j] -= counts_to_commit
                    deficit = 0
                    if self._counters[j].total() == 0:  # => Don't bother filling this subcounter. It will be cleaned up later on.
                        j += 1

            # We are done handling counter i. It has left a trail of any amount of empty counters. If index j is not out-of-range, its counter is non-empty, and we handle it as i next.
            i = j

        # Finally, clean up the internal state.
        self._garbageCollect()
        assert total_before_defragmenting == sum(self.subcounterSizes())  # This checks for two bugs: (1) if the total_before_defragmenting was equal to sum(subcounters), this checks whether the subcounters lost any elements, and (2) if the subcounters did not lose any elements, it checks whether the total inside the main counter was equal to the total in the subcounters.

    def sampleSubcounter(self, subcounter_idx: int, n: int) -> Counter[K]:
        """
        Sample according to the distribution in the next counter, making sure to not sample more of an element than exists (which can only happen once).
        """
        assert n >= 0

        counter = self._counters[subcounter_idx]
        keys = list(counter)
        counts_available = [counter[key] for key in keys]
        counts_to_return = [0 for _ in counts_available]
        p = np.array(counts_available) / sum(counts_available)
        while n:
            new_samples = self._rng.multinomial(n=n, pvals=p)  # We don't need to sample individual elements; we can sample counts immediately.
            for idx in range(len(keys)):
                # Find where these counts violate what we have available. Those elements are depleted.
                sampled_count = new_samples[idx]
                if sampled_count >= counts_available[idx]:
                    sampled_count = counts_available[idx]
                    p[idx] = 0

                # Reduce available elements and increase the elements to commit.
                counts_available[idx] -= sampled_count  # We don't change the probabilities of sampling idx, but we do reduce how much can still be sampled before the probability is set to 0.
                counts_to_return[idx] += sampled_count
                n -= sampled_count

            denom = p.sum()
            if denom == 0.0:
                break
            p /= denom

        return Counter({keys[i]: counts_to_return[i] for i in range(len(keys))})

    def copy(self) -> Self:
        out = ChainedCounter(self._max_size, self._seed)

        # Fill the internal dictionary of `out` using the internal dictionary of `self`.
        for k,v in self.items():
            out[k] += v

        # Copy over the exact state of the subcounters, which are equivalent to the internal dictionary but not the same.
        out._counters = [counter.copy() for counter in self._counters]
        out._garbageCollect()
        return out

    def _regenerate(self) -> Iterator[K]:
        """
        Output exactly the elements that are inside this counter, subcounter by subcounter, sampling the current
        subcounter according to the distribution over unique values as exhibited by the elements in the starting state.

        Note: This is VERY slow. You should basically avoid this in all implementations. Often you don't actually
              need to generate individual elements, but rather just need to sample counts multinomially.
        """
        for counter in self._counters:
            # Get distribution
            keys          = list(counter.keys())
            counts        = [counter[k] for k in keys]
            total_count   = sum(counts)  # == counter.total()
            probabilities = np.array([c/total_count for c in counts])

            # Sample from the distribution until you run out of samples.
            n_remaining_keys = len(keys)
            for _ in range(total_count):
                key_idx = self._rng.choice(len(keys), p=probabilities)
                if n_remaining_keys == 1:
                    for _ in range(counts[key_idx]):
                        yield keys[key_idx]
                    break

                yield keys[key_idx]

                counts[key_idx] -= 1
                if counts[key_idx] == 0:  # This key index must never be selected again. Renormalise the remaining probabilities.
                    probabilities[key_idx] = 0
                    probabilities = probabilities / np.sum(probabilities)
                    n_remaining_keys -= 1

    def _garbageCollect(self):
        """Remove subcounters that contain no elements."""
        self._counters = [c for c in self._counters if c.total() > 0]
        self._size_of_last_counter = self._counters[-1].total() if self._counters else self._max_size

    def get(self, key: K, default=None):
        # For some unknown reason, Counter's implementation of .get() does not use .__getitem__(), which this class
        # originally re-implemented. To support .get(), and by extension .update() -- which uses .get() rather than
        # using .__getitem__() -- we had to override it.
        # I'm pretty sure this is now obsolete since we no longer override .__getitem__().
        if key in self:
            return self[key]
        else:
            return default
