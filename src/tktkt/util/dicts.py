from typing import TypeVar, Dict, List, Generic, Callable, Union, Iterator, Tuple
from collections import OrderedDict, Counter
import numpy as np
import numpy.random as npr

K = TypeVar("K")
V = TypeVar("V")
Number = TypeVar("Number", bound=Union[int,float])


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

    def __init__(self, max_subcounter_size: int, seed: int=0):
        super().__init__()  # Because this is a subclass of Counter, the super class has a "master data structure".
        self._counters = []
        self._size_of_last_counter = max_subcounter_size

        self._max_size = max_subcounter_size
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
        is that it will add  value - object[key]  to the subcounters, not value.)
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

        sum_counter = ChainedCounter(self._max_size, seed=self._rng.integers(0, 1_000_000))

        # Add this counter's elements first
        for counter in self._counters:
            for k,v in counter.items():
                sum_counter[k] += v

        # Add the other counter's elements, as if they all came after. This is an important assumption.
        for element in other._regenerate():
            sum_counter[element] += 1

        return sum_counter

    def repack(self, sort_subcounters_first: bool=False):
        """
        Re-pack values inside this counter so that any gaps left by popping from the subcounters are filled up again.
        """
        if not self._counters:
            return

        if sort_subcounters_first:
            self._counters.sort(key=lambda c: -c.total())  # Biggest subcounter first.

        i = 0
        while i < self.subcounterAmount():
            deficit = self._max_size - self._counters[i].total()
            j = i + 1  # Steal items from the next counter.
            while deficit and j < self.subcounterAmount():
                next_counter = self._counters[j]
                available = next_counter.total()
                if deficit > available:  # The next counter can't fill the gap. Copy in its entirety. No need to sample.
                    self._counters[i] += next_counter
                    deficit -= available
                    j += 1
                else:  # Sample according to the distribution in the next counter, making sure to not sample more of an element than exists (which can only happen once).
                    keys   = list(next_counter)
                    counts_available = [next_counter[key] for key in keys]
                    counts_to_commit = [0 for _ in counts_available]
                    p = np.array(counts_available) / sum(counts_available)
                    while deficit:
                        new_samples = self._rng.multinomial(n=deficit, pvals=p)  # We don't need to sample individual elements; we can sample counts immediately.
                        for idx in range(len(keys)):
                            # Find where these counts violate what we have available. Those elements are depleted.
                            sampled_count = new_samples[idx]
                            if sampled_count >= counts_available[idx]:
                                sampled_count = counts_available[idx]
                                p[idx] = 0

                            # Reduce available elements and increase the elements to commit.
                            counts_available[idx] -= sampled_count  # We don't change the probabilities of sampling idx, but we do reduce how much can still be sampled before the probability is set to 0.
                            counts_to_commit[idx] += sampled_count
                            deficit -= sampled_count

                        denom = p.sum()
                        if denom == 0.0:
                            break
                        p /= denom

                    counts_to_commit = Counter({keys[i]: counts_to_commit[i] for i in range(len(keys))})
                    self._counters[i] += counts_to_commit
                    self._counters[j] -= counts_to_commit
                    if self._counters[j].total() == 0:
                        j += 1

            # We are done handling counter i. It has left a trail of any amount of empty counters. If index j is valid, it is non-empty.
            i = j

        # Finally, clean up the internal state.
        self._garbageCollect()

    def _regenerate(self) -> Iterator[K]:
        """
        Output exactly the elements that are inside this counter, subcounter by subcounter, sampling the current
        subcounter according to the distribution over unique values as exhibited by the elements in the starting state.

        FIXME: This is VERY slow. You should basically avoid this in all implementations. It happens once in
               TkTkT (analyseAccessors) and therefore it works way slower than it could. Often you don't actually
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
                if n_remaining_keys == 1:
                    for _ in range(counts[0]):
                        yield keys[0]
                    break

                key_idx = self._rng.choice(len(keys), p=probabilities)
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
