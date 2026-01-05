"""
For general-purpose classes that have @abstractmethods.
"""
from abc import abstractmethod, ABC
from typing import Iterable, TypeVar, Mapping, Generic, Callable, Optional

from nltk.lm import counter
from typing_extensions import Self
from collections import Counter
from pathlib import Path

K = TypeVar("K")
V = TypeVar("V")


class ExtensibleMapping(Mapping[K,V], ABC):  # (No longer used since this assumes specials are added AFTER the vocab exists, not BEFORE.)

    def __init__(self):
        self.hardcoded: dict[K,V] = dict()

    @abstractmethod
    def _get(self, key: K) -> V:
        pass

    @abstractmethod
    def _keys(self) -> Iterable[K]:
        pass

    @abstractmethod
    def _values(self) -> Iterable[V]:
        pass

    @abstractmethod
    def _items(self) -> Iterable[tuple[K, V]]:
        pass

    ###############################################

    def get(self, key: K) -> V:
        try:
            return self.hardcoded[key]
        except:
            try:
                return self._get(key)
            except:
                return None

    def set(self, key: K, value: V):
        original_value = value
        while value in set(self.values()):
            value += 1
        self.hardcoded[key] = value
        if value != original_value:
            import warnings
            warnings.warn(f"Requested to set value {original_value} for key {key}, but was increased to {value} due to collisions.")

    def keys(self) -> Iterable[K]:
        yield from self.hardcoded.keys()
        yield from self._keys()

    def values(self) -> Iterable[V]:
        yield from self.hardcoded.values()
        yield from self._values()

    def items(self) -> Iterable[tuple[K,V]]:
        yield from self.hardcoded.items()
        yield from self._items()

    def __getitem__(self, key: K) -> V:
        return self._get(key)

    def __len__(self) -> int:
        count = 0
        for _ in self:
            count += 1
        return count

    def __iter__(self):
        yield from self.keys()


class Cacheable(ABC):
    """
    Something that can be serialised and deserialised, and in addition can check if it exists at a given path.
    """

    @classmethod
    @abstractmethod
    def exists(cls, cache_path: Path) -> bool:
        """
        Whether the files that should be computed once already exist,
        NOT necessarily whether an empty container exists wherein to put them.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, cache_path: Path) -> Self:
        pass

    @abstractmethod
    def store(self, cache_path: Path):
        pass


C = TypeVar("C", bound=Cacheable)

class Cache(Generic[C], ABC):
    """
    Can skip its computation using a cache.
    """

    def __init__(self, disable_cache: bool=False):
        self._disable_cache = disable_cache

    @abstractmethod
    def _cacheType(self) -> type[C]:
        """Declares the class which knows how to store and load the data we want to cache."""
        pass

    @abstractmethod
    def _cacheFinalise(self, loaded: C) -> C:
        """Post-processing applied to cached data once it has been loaded. (Is NOT applied when it is computed.)"""
        pass

    @abstractmethod
    def _cachePath(self, unambiguous_cache_identifier: str) -> Path:
        """
        Constructs (and possibly creates) the main folder or file used for this cache.
        :param unambiguous_cache_identifier: unique identifier by which to associate the results. Two runs with the same identifier should be indistinguishable.
        """
        pass

    def _cacheRun(self, identifier: str, imputation: Callable[[],C]) -> C:
        if self._disable_cache:  # Bypass loading and storing
            result = imputation()
        else:
            cache_path  = self._cachePath(identifier)
            cache_class = self._cacheType()
            if cache_class.exists(cache_path):
                result = cache_class.load(cache_path)
                result = self._cacheFinalise(result)
            else:
                result = imputation()
                result.store(cache_path)
        return result

    # A status is a stored flag. Useful when you have a caching mechanism with multiple stages and you need to know which stage you are in.

    def _cacheStatusRead(self, cache_path: Path) -> Optional[str]:
        if not cache_path.is_dir():  # If is_dir is false, either the path is a file path or it is a directory that doesn't exist.
            cache_path = cache_path.parent
        status_path = cache_path / ".status"
        if not status_path.exists():  # In both of the aforementioned cases, the status won't be found if and only if it doesn't exist.
            return None
        with open(status_path, "r", encoding="utf-8") as handle:
            return handle.readline().strip()

    def _cacheStatusWrite(self, cache_path: Path, status: str) -> Optional[str]:
        old_status = self._cacheStatusRead(cache_path)

        if not cache_path.is_dir():
            cache_path = cache_path.parent
        status_path = cache_path / ".status"
        with open(status_path, "w", encoding="utf-8") as handle:
            handle.write(status)

        return old_status

    def _cacheStatusClear(self, cache_path: Path) -> Optional[str]:
        old_status = self._cacheStatusRead(cache_path)
        if not cache_path.is_dir():
            cache_path = cache_path.parent
        status_path = cache_path / ".status"
        status_path.unlink(missing_ok=True)
        return old_status


class SimpleCacheableDataclass(Cacheable):
    """For providing a load/store to simple dataclasses."""

    @classmethod
    def _stem(cls):
        from .strings import convertCase, Case
        return convertCase(cls.__name__.replace("Return", ""), from_case=Case.PASCAL, to_case=Case.SNAKE)

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return (cache_path / (cls._stem() + ".json")).exists()

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        import json
        from dacite import from_dict
        with open(cache_path / (cls._stem() + ".json"), "r", encoding="utf-8") as handle:
            return from_dict(cls, json.load(handle))

    def store(self, cache_path: Path):
        import json
        from dataclasses import asdict
        with open(cache_path / (self._stem() + ".json"), "w", encoding="utf-8") as handle:
            return json.dump(asdict(self), handle, indent=4)


class CacheableCounter(Cacheable, Counter[K]):
    """For providing a load/store to simple counters."""

    @classmethod
    def _stem(cls):
        from .strings import convertCase, Case
        return convertCase(cls.__name__.replace("Return", ""), from_case=Case.PASCAL, to_case=Case.SNAKE)

    @classmethod
    @abstractmethod
    def _stringToKey(cls, s: str) -> K:
        pass

    @classmethod
    @abstractmethod
    def _keyToString(cls, k: K) -> str:
        pass

    @classmethod
    def exists(cls, cache_path: Path) -> bool:
        return (cache_path / (cls._stem() + ".tsv")).exists()

    @classmethod
    def load(cls, cache_path: Path) -> Self:
        counter = cls()
        with open(cache_path / (cls._stem() + ".tsv"), "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                part1, part2 = line.split("\t")
                counter[cls._stringToKey(part1)] = int(part2)
        return counter

    def store(self, cache_path: Path):
        with open(cache_path / (self._stem() + ".tsv"), "w", encoding="utf-8") as handle:
            for k, v in self.items():
                handle.write(f"{self._keyToString(k)}\t{v}\n")
