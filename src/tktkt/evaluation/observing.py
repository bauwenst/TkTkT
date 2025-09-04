"""
The Observable/Observer system was designed to solve two related problems:
    (1) Having to iterate over the same corpus many times when computing different metrics, often in multiple runs;
    (2) Having to do the work of tokenisation again every time when iterating over the corpus.

The idea is that you iterate over a corpus once. If any metric needs tokens, you tokenise exactly once, and then you
distribute your tokens to all the metrics that need them. The results computed by those metrics can then be passed to
other metrics, and so on.

It's kind of a pub/sub or callback or recursive map() system.
Observables communicate with Observers through method calls. Observers communicate with Observables using exceptions.
"""
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Iterator, Any, Callable, Tuple, Union
from pathlib import Path

import traceback

from ..interfaces import Preprocessor
from ..interfaces.tokeniser import Tokeniser
from ..util.dicts import optionalDataclassToDict
from ..util.iterables import dunion
from ..util.types import NamedIterable, Tokens

Received = TypeVar("Received")
Sent     = TypeVar("Sent")


class ObserverEarlyExit(Exception):
    pass


class Observer(ABC, Generic[Received]):
    """
    Receives data and processes it.

    Note 1: Is not aware of who gives it data. That does not just mean "doesn't know which Observable gives it data",
            but also "doesn't even know an Observable gives it data". Thus, making somesample an Observer does not mean
            that it always needs to see the data it receives in the context of an Observable's stream.

    Note 2: By default, does not send its results to other Observers (which is fine for e.g. numerical metrics).
    """

    @abstractmethod
    def _initialise(self, global_run_identifier: str):
        """
        Reset relevant fields at the start of a run, and store (or pass down) an identifier for the run that is unique across projects.

        :raise ObserverEarlyExit: When it is noticed that running this observer is unnecessary.
        """
        pass

    @abstractmethod
    def _receive(self, sample: Received, weight: float):
        """
        :param weight: Amount of calls to _receive() this sample should be equivalent to.
        :raise ObserverEarlyExit: When it is noticed that there is no point continuing to receive samples.
        :raise Exception: Can also raise any other exception when things break.
        """
        pass

    @abstractmethod
    def _finish(self):
        pass


class AppendToListObserver(Observer[Any]):

    def __init__(self, list_to_append_to: list):
        self._list_reference = list_to_append_to

    def _initialise(self, global_run_identifier: str):
        pass

    def _receive(self, sample: Any, _):
        self._list_reference.append(sample)

    def _finish(self):
        pass


class DataclassCollectorObserver(Observer[Any]):
    """
    Observer meant to be put at various points in the hierarchy, to collect dictionaries/dataclasses that are
    supposed to be saved together as one row in a CSV file. The user is then responsible for separating rows.
    """

    def __init__(self):
        self._current_list    = []
        self._completed_lists = []

    def addMetadata(self, metadata: dict):
        self._receive(metadata, 1)

    def fence(self):
        """Finish the current collection and start a new collection."""
        self._completed_lists.append(self._current_list)
        self._current_list = []

    def assemble(self) -> List[dict]:
        """For each collection that has been fenced, pool together all the dataclasses/dictionaries."""
        return [dunion(map(optionalDataclassToDict, dicts)) for dicts in self._completed_lists]

    def _initialise(self, global_run_identifier: str):
        pass

    def _receive(self, sample: Any, _):
        self._current_list.append(sample)

    def _finish(self):
        # Explicitly does nothing, because this method will be called at multiple points in the hierarchy.
        pass


class PrintingObserver(Observer[Any]):

    def _initialise(self, global_run_identifier: str):
        pass

    def _receive(self, sample: Any, _):
        print(sample)

    def _finish(self):
        pass


class PrintIfContainsToken(Observer[Tokens]):
    def __init__(self, print_if_contains: set[str]):
        self._target_set = print_if_contains

    def _initialise(self, global_run_identifier: str):
        pass

    def _receive(self, sample: Tokens, _):
        for token in sample:  # O(N)
            if token in self._target_set:  # O(1), rather than O(T), which you'd get by iterating.
                print(sample)
                break

    def _finish(self):
        pass


class Observable(Generic[Sent]):
    """
    Knows observers and sends data to them.
    By default, has no abstract methods to implement; .
    """

    def __init__(self, observers: List[Observer[Sent]]):
        assert observers, "An observable must have at least one observer."
        self.observers = observers
        self.done = []
        self.dead = []

    def _callObservers(self, callable: Callable[[Observer],None]):
        """
        :raise ObserverEarlyExit: When AFTER this call (we never check BEFORE!), this observable has no more observers.
        """
        done = []
        dead = []

        # Send sample to all observers.
        for observer in self.observers:
            try:
                callable(observer)
            except ObserverEarlyExit:
                print("Observer finished early and moved to the done queue:", observer.__class__.__name__)
                done.append(observer)
            except:
                print("Observer failed and moved to the dead queue:", observer.__class__.__name__)
                print(traceback.format_exc())
                dead.append(observer)

        # Triage observers if they had exceptions. (This is not part of the main loop to prevent having to copy self.observers for each sample to be able to modify self.observers.)
        for observer in done:
            self.done.append(observer)
            self.observers.remove(observer)
            if not self.observers:  # Note that this check is only performed when self.observers is actually modified.
                raise ObserverEarlyExit()
        for observer in dead:
            self.dead.append(observer)
            self.observers.remove(observer)
            if not self.observers:
                raise ObserverEarlyExit()

    def _initialiseObservers(self, global_run_identifier: str):  # This method only to be called by users of this class, not by the class itself.
        self._callObservers(lambda observer: observer._initialise(global_run_identifier))

    def _send(self, sample: Sent, weight: float):
        self._callObservers(lambda observer: observer._receive(sample, weight))

    def _finishObservers(self):  # This method only to be called by users of this class, not by the class itself.
        for observer in self.observers + self.done:
            try:
                observer._finish()
            except ObserverEarlyExit:
                pass
            except:
                print("Observer failed while finishing:", observer.__class__.__name__)
                print(traceback.format_exc())


class ObservableRoot(Observable[Sent]):
    """Observable with a method that opens a constant stream of samples."""

    @abstractmethod
    def _stream(self) -> Iterator[Tuple[Sent,float]]:
        pass

    @abstractmethod
    def _globalRunIdentifier(self) -> str:
        """Method used to inform observers which of their caches to use."""
        pass

    def run(self):
        # Initialise
        skip_everything = False
        try:
            self._initialiseObservers(self._globalRunIdentifier())
        except Exception as e:
            skip_everything = True

        # Run
        if not skip_everything:
            for sample, weight in self._stream():  # If self._stream() throws an error, we let it crash the method.
                try:
                    self._send(sample, weight)
                except:  # _send() raises an exception when all observers have stopped. Hence, move on to finish().
                    break

        # Finish
        self._finishObservers()


class ObservableIterable(ObservableRoot[Sent]):
    """
    Observable that calls its send() method itself by iterating an iterable.
    Equivalently, a reverse iterable, which rather than client code iterating over it, iterates over client code.
    """

    def __init__(self, iterable: Union[NamedIterable[Sent], NamedIterable[Tuple[Sent,float]]], already_contains_weights: bool=False, observers: List[Observer[Sent]]=None):
        super().__init__(observers=observers)
        self.iterable = iterable
        self._is_tuple_with_weight = already_contains_weights

    def _globalRunIdentifier(self) -> str:  # TODO: You're going to want a constructor that takes an extra name, because just the corpus name is obviously not enough.
        return self.iterable.name

    def _stream(self):
        if self._is_tuple_with_weight:
            yield from self.iterable
        else:
            for sample in self.iterable:
                yield sample, 1


class ObservableObserver(Observer[Received], Observable[Sent]):
    """
    An observer which itself is observed.
    """
    def _initialiseAsObserver(self, identifier: str):
        """Initialise this object as if it is just an observer, without dependent observers."""
        pass

    def _finishAsObserver(self):
        """Finish this object as if it is just an observer, without dependent observers."""
        pass

    def _initialise(self, global_run_identifier: str):
        self._initialiseObservers(global_run_identifier)  # First initialise the observers, since they can't see this object anyway.
        self._initialiseAsObserver(global_run_identifier)

    def _finish(self):
        self._finishAsObserver()  # Object as an observer. First finish yourself, since you may still want to send a sample to your observers.
        self._finishObservers()


class ObservablePreprocessor(ObservableObserver[str,str]):

    def __init__(self, preprocessor: Preprocessor, observers: List[Observer[str]]):
        super().__init__(observers=observers)
        self.preprocessor = preprocessor

    def _receive(self, sample: str, weight: float):
        for pretoken in self.preprocessor.do(sample):
            self._send(pretoken, weight)


class ImmediatelyObservableObserver(ObservableObserver[Received,Sent]):
    """
    Special kind of observable observer which outputs exactly one sample whenever it receives a sample.
    """
    
    @abstractmethod
    def _transit(self, sample: Received, weight: float) -> Sent:
        pass

    def _receive(self, sample: Sent, weight: float):
        self._send(self._transit(sample, weight), weight)


class ObservableTokeniser(ImmediatelyObservableObserver[str,Tokens]):

    def __init__(self, tokeniser: Tokeniser, observers: List[Observer[Tokens]]):
        super().__init__(observers=observers)
        self.tokeniser = tokeniser

    def _transit(self, sample: str, _) -> Tokens:
        return self.tokeniser.prepareAndTokenise(sample)


class ObservableFunction(ImmediatelyObservableObserver[Received,Sent]):

    def __init__(self, f: Callable[[Received,float],Sent], observers: List[Observer[Sent]]):
        super().__init__(observers=observers)
        self._function = f

    def _transit(self, sample: Received, weight: float) -> Sent:
        return self._function(sample, weight)


_Received2 = TypeVar("_Received2")
class SplitObserver(Observer[Tuple[Received,_Received2]]):
    """
    Really, this is an ObservableObserver (it is an observer because it can be sent stuff, and it is an observable because
    it can send stuff), but we can't extend the traditional Observable interface because it expects only one list of observers.
    """

    def __init__(self, observers1: List[Observer[Received]], observers2: List[Observer[_Received2]]):
        self._observable1 = Observable(observers1)  # We use an observable (i.e. something without a ._receive() method) because we merely need to distribute across the observers, without extra behaviour. Basically equivalent to ImmediatelyObservableObserver with ._transit() being the identity function.
        self._observable2 = Observable(observers2)

    def _initialise(self, global_run_identifier: str):
        self._observable1._initialiseObservers(global_run_identifier)
        self._observable2._initialiseObservers(global_run_identifier)

    def _receive(self, sample: Tuple[Received,_Received2], weight: float):
        left, right = sample
        self._observable1._send(left,  weight)  # ._receive() loop across the observers.
        self._observable2._send(right, weight)

    def _finish(self):
        self._observable1._finishObservers()
        self._observable2._finishObservers()


class FinallyObservableObserver(ObservableObserver[Received, Sent]):
    """
    Special kind of ObservableObserver which only outputs something when its Observable is FINISHED.
    Its observers need to be sent the result before they are finished too.

    This is the one type of Observer that has caching.
    """

    def __init__(self, cache_disambiguator: str= "", disable_cache: bool=False, observers: List[Observer[Sent]]=None):
        """
        :param cache_disambiguator: If you have two observers that would use the same cache given the same run identifier,
                                    this argument allows separating their two caches.
        """
        super().__init__(observers=observers)
        self._disable_cache = disable_cache
        self._stored_global_run_identifier = ""
        self._disambiguation_identifier = cache_disambiguator

    def _initialise(self, global_run_identifier: str):
        # Init your observers and init yourself. This can throw an exception.
        super()._initialise(global_run_identifier)

        # Caching.
        self._stored_global_run_identifier = global_run_identifier
        if not self._disable_cache and self._cacheExists(self._cachePath(self._cacheIdentifier())):
            raise ObserverEarlyExit()

    @abstractmethod
    def _compute(self) -> Sent:
        pass

    def _cacheIdentifier(self) -> str:
        return self._stored_global_run_identifier + ("_" + self._disambiguation_identifier if self._disambiguation_identifier else "")

    @abstractmethod
    def _cachePath(self, unambiguous_cache_identifier: str) -> Path:  # Can be a folder or a file.
        pass

    def _cacheExists(self, cache_path: Path) -> bool:
        return cache_path.exists()  # Not always sufficient, but the default is that it is.

    @abstractmethod
    def _cacheLoad(self, cache_path: Path) -> Sent:
        pass

    @abstractmethod
    def _cacheStore(self, cache_path: Path, result: Sent):
        pass

    def _finishAsObserver(self):
        if self._disable_cache:
            result = self._compute()
        else:
            cache_path = self._cachePath(self._cacheIdentifier())
            if self._cacheExists(cache_path):
                result = self._cacheLoad(cache_path)
            else:
                result = self._compute()
                self._cacheStore(cache_path, result)

        self._send(result, 1)
        # Only after THIS will its own observers be finished.


def evaluateTokeniser(corpus: NamedIterable[str], tokeniser: Tokeniser, token_consumers: List[Observer[Tokens]]):
    """
    Functional shorthand for the object-oriented Observable/Observer approach in case where you want to use a corpus
    and compute metrics over the tokeniser's token outputs.
    """
    return ObservableIterable(
        iterable=corpus,
        observers=[
            ObservableTokeniser(
                tokeniser=tokeniser,
                observers=token_consumers
            )
        ]
    )


def evaluateTokeniserOnWords(corpus: NamedIterable[str], word_preprocessor: Preprocessor, tokeniser: Tokeniser, token_consumers: List[Observer[Tokens]]):
    return ObservableIterable(
        iterable=corpus,
        observers=[
            ObservablePreprocessor(
                preprocessor=word_preprocessor,
                observers=[
                    ObservableTokeniser(
                        tokeniser=tokeniser,
                        observers=token_consumers
                    )
                ]
            )
        ]
    )
