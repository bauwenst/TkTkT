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

import numpy.random as npr
import traceback
import colorama

from ..interfaces import Preprocessor
from ..interfaces.tokeniser import Tokeniser
from ..util.dicts import optionalDataclassToDict
from ..util.iterables import dunion
from ..util.types import NamedIterable, Tokens, HoldoutState
from ..util.strings import indent, underscoreIfNotEmpty

Received = TypeVar("Received")
Sent     = TypeVar("Sent")


def formatException(exception_string: str) -> str:
    return indent(1, f"{colorama.Fore.YELLOW}{exception_string}{colorama.Fore.RESET}")


def formatEarlyExit(msg: str) -> str:
    return indent(1, f"{colorama.Fore.CYAN}{msg}{colorama.Fore.RESET}")


def formatExit(msg: str) -> str:
    return indent(1, f"{colorama.Fore.GREEN}{msg}{colorama.Fore.RESET}")


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
    supposed to be saved together as one row in a CSV file.

    The user decides when a new row is started.
    """

    class DataclassSuffixer(Observer[Any]):
        """
        Shim put in front of a DataclassCollectorObserver which adds a suffix to each key / field name that is received.
        """

        def __init__(self, master: "DataclassCollectorObserver", suffix: str):
            self._master = master
            self._suffix = suffix

        def _initialise(self, global_run_identifier: str):
            pass

        def _receive(self, sample: Any, _):
            self._master._receive({f"{k}_{self._suffix}": v for k,v in optionalDataclassToDict(sample).items()}, _)

        def _finish(self):
            pass

    def withSuffix(self, suffix: str) -> DataclassSuffixer:
        """Add a suffix to the end of the fields of the received dataclasses."""
        return DataclassCollectorObserver.DataclassSuffixer(self, suffix)

    ####################################################################################################################

    def __init__(self, fence_on_assemble: bool=True):
        self._current_list    = []
        self._completed_lists = []
        self._fence_on_assemble = fence_on_assemble

    def addMetadata(self, metadata: dict):
        """Add metadata to the current row."""
        self._receive(metadata, 1)

    def fence(self):
        """Finish the current collection and start a new collection."""
        self._completed_lists.append(self._current_list)
        self._current_list = []

    def assemble(self) -> List[dict]:
        """For each collection that has been fenced, pool together all the dataclasses/dictionaries."""
        if self._fence_on_assemble and self._current_list:
            self.fence()
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


class ObservableMeta(ABC, Generic[Sent]):
    """
    Knows observers and sends data to them.
    """
    @abstractmethod
    def _initialiseObservers(self, global_run_identifier: str):
        pass

    @abstractmethod
    def _send(self, sample: Sent, weight: float):
        pass

    @abstractmethod
    def _finishObservers(self):
        pass

    @abstractmethod
    def anyObserversAlive(self) -> bool:
        pass

    @abstractmethod
    def observerStatusReport(self) -> str:
        """
        Formatted string that contains, at least, one line for each of the observers associated with this observable,
        with some information about its status.
        """
        pass


class Observable(ObservableMeta[Sent]):
    """
    Simplest implementation of the ObservableMeta's methods using a single list of observables, which get sent
    samples of the type given by the type parameter.
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
                print(formatEarlyExit(f"Observer finished early: {observer.__class__.__name__}"))
                done.append(observer)
            except:
                print(formatException(f"Observer failed: {observer.__class__.__name__}"))
                print(formatException(traceback.format_exc()))
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
                print(f"Observer failed while finishing: {observer.__class__.__name__}")
                print(formatException(traceback.format_exc()))

    def anyObserversAlive(self) -> bool:
        return len(self.observers) > 0

    def observerStatusReport(self) -> str:
        def getObserverReport(observer: Observer) -> str:
            if observer in self.observers:
                prefix = "✅"
            elif observer in self.done:
                if isinstance(observer, ObservableMeta) and not observer.anyObserversAlive():
                    prefix = "💤"
                else:
                    prefix = "💾"
            elif observer in self.dead:
                prefix = "❌"
            else:
                raise RuntimeError()

            report = f"{prefix} {observer.__class__.__name__}"
            if isinstance(observer, ObservableMeta):
                report += "\n" + indent(1, observer.observerStatusReport())
            return report

        return "\n".join(map(getObserverReport, self.observers + self.done + self.dead))


class ObservableRoot(Observable[Sent]):
    """Observable with a method that opens a constant stream of samples."""

    def __init__(self, cache_disambiguator: str):
        """
        :param cache_disambiguator: disambiguates runs whose root instances are otherwise identical.
                                    TODO: A smarter way to do this would be to instead have nodes in the tree
                                          add on to the name of the root. Because indeed, the only thing that CAN
                                          change in the experiment if the root node is the same but the cache is not,
                                          is any node between the two, e.g. the tokeniser, which the user is now asked to identify in the disambiguator manually.
        """
        self._cache_disambiguator = cache_disambiguator

    @abstractmethod
    def _stream(self) -> Iterator[Tuple[Sent,float]]:
        pass

    def _globalRunIdentifier(self) -> str:
        """Informs observers which of their caches to use."""
        return self._nodeIdentifier() + underscoreIfNotEmpty(self._cache_disambiguator)

    @abstractmethod
    def _nodeIdentifier(self) -> str:
        """Disambiguates instances of this root."""
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

        print(formatExit("ObservableRoot finished running."))
        print(formatExit(self.observerStatusReport()))


class ObservableIterable(ObservableRoot[Sent]):
    """
    Observable that calls its send() method itself by iterating an iterable.
    Equivalently, a reverse iterable, which rather than client code iterating over it, iterates over client code.
    """

    def __init__(self, experiment_id: str, iterable: Union[NamedIterable[Sent], NamedIterable[Tuple[Sent,float]]], already_contains_weights: bool=False, observers: List[Observer[Sent]]=None):
        super().__init__(cache_disambiguator=experiment_id, observers=observers)
        self.iterable = iterable
        self._is_tuple_with_weight = already_contains_weights

    def _nodeIdentifier(self) -> str:
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
        # When you have dependent observers, it is more important to check if THEY can initialise than to check if YOU can.
        # If all your observers signal to you that you don't need to do any work, you should not even initialise.
        self._initialiseObservers(global_run_identifier)  # Note also that since the observers can't see this object, they cannot crash due to lack of initialisation here.
        self._initialiseAsObserver(global_run_identifier)  # This line will only run if there are observers left.

    def _finish(self):
        if self.anyObserversAlive():  # You should ONLY finish if you ran to completion, i.e. if they stopped sending you samples because there were no more samples. You can figure this out by checking if you have alive observers (which is only a subset of the observers that need to get a finishing signal).
            self._finishAsObserver()  # First finish yourself, since you may still want to send a sample to your observers.
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


class ObservableFilter(ObservableObserver[Received,Received]):

    def __init__(self, predicate: Callable[[Received],bool], observers: List[Observer[Received]]):
        super().__init__(observers=observers)
        self._predicate = predicate

    def _receive(self, sample: Received, weight: float):
        if self._predicate(sample):
            self._send(sample, weight)


class HoldoutObserver(ObservableFilter[Received]):

    def __init__(self, holdout: HoldoutState, test_split: bool, observers: List[Observer[Received]]):
        super().__init__(observers=observers, predicate=lambda sample: holdout.decide() != test_split)
        self._holdout = holdout

    def _initialiseAsObserver(self, identifier: str):
        self._holdout.reset()


_Received2 = TypeVar("_Received2")
class SplitObserver(Observer[Tuple[Received,_Received2]], ObservableMeta[Tuple[Received,_Received2]]):
    """
    Really, this is an ObservableObserver (it is an observer because it can be sent stuff, and it is an observable because
    it can send stuff), but we can't extend the traditional Observable interface because it expects only one list of observers.
    """

    def __init__(self, observers1: List[Observer[Received]], observers2: List[Observer[_Received2]]):
        self._observable1 = Observable(observers1)  # We use an observable (i.e. something without a ._receive() method) because we merely need to distribute across the observers, without extra behaviour. Basically equivalent to ImmediatelyObservableObserver with ._transit() being the identity function.
        self._observable2 = Observable(observers2)

    # Observable methods (i.e. methods as something that has observers)

    def _logicalAndExceptions(self, call_observers1: Callable[[],None], call_observers2: Callable[[],None]):
        """
        Only throws an exception if both functions throw an exception.
        """
        try:  # Try the first function.
            call_observers1()
        except ObserverEarlyExit:  # If first function fails, do not protect the second function.
            call_observers2()
        else:  # If first function succeeds successfully, always protect the second function.
            try:
                call_observers2()
            except ObserverEarlyExit:
                pass

    def _initialiseObservers(self, global_run_identifier: str):
        self._logicalAndExceptions(
            lambda: self._observable1._initialiseObservers(global_run_identifier),
            lambda: self._observable2._initialiseObservers(global_run_identifier)
        )

    def _send(self, sample: Tuple[Received,_Received2], weight: float):
        left, right = sample
        self._logicalAndExceptions(
            lambda: self._observable1._send(left,  weight),  # ._receive() loop across the observers.
            lambda: self._observable2._send(right, weight)
        )

    def _finishObservers(self):
        self._logicalAndExceptions(
            lambda: self._observable1._finishObservers(),
            lambda: self._observable2._finishObservers()
        )

    def anyObserversAlive(self) -> bool:
        return self._observable1.anyObserversAlive() or self._observable2.anyObserversAlive()

    def observerStatusReport(self) -> str:
        return self._observable1.observerStatusReport() + "\n" + self._observable2.observerStatusReport()

    # Observer methods

    def _initialise(self, global_run_identifier: str):
        self._initialiseObservers(global_run_identifier)

    def _receive(self, sample: Tuple[Received,_Received2], weight: float):
        self._send(sample, weight)

    def _finish(self):
        self._finishObservers()


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
        return self._stored_global_run_identifier + underscoreIfNotEmpty(self._disambiguation_identifier)

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


class WirelessObserverConnection(Generic[_Received2]):
    """
    Proxy object so that wireless receivers can be instantiated without needing a reference to the sender which is
    instantiated in the same tree.
    """

    def __init__(self):
        self._sender: WirelessSplittingObserver[Any,_Received2] = None
        self._pending_receivers: List[WirelessRecombiningObserver] = []

    def _connectSender(self, splitter: "WirelessSplittingObserver[Any,_Received2]"):
        self._sender = splitter
        for receiver in self._pending_receivers:
            self._connectReceiver(receiver)
        self._pending_receivers = []

    def _connectReceiver(self, receiver: "WirelessRecombiningObserver"):
        if self._sender is None:
            self._pending_receivers.append(receiver)
        else:
            receiver._registerServer(self._sender)
            self._sender._registerReceiver()


class WirelessSplittingObserver(ObservableObserver[Tuple[Received,_Received2],Received]):
    """
    Takes in tuple samples (e.g. text and frequency) and sends the first half to its observers, while the second
    half is stored in a buffer until all recombiners on the same connection as the splitter have requested it.
    """
    def __init__(self, connection: WirelessObserverConnection[_Received2], observers: List[Observer[Received]]):
        super().__init__(observers=observers)
        self._buffer: List[_Received2]      = []
        self._remaining_requests: List[int] = []
        self._min_index   = 0

        self._n_receivers = 0
        connection._connectSender(self)

    def _registerReceiver(self):
        self._n_receivers += 1

    def _receive(self, sample: Tuple[Received,_Received2], weight: float):
        left, right = sample
        self._buffer.append(right)  # Important that this comes before the call to send, because that call will likely trigger all the requests to get the value back.
        self._remaining_requests.append(self._n_receivers)
        self._send(left, weight)

    def _request(self, index: int) -> _Received2:
        result = self._buffer[index - self._min_index]
        self._remaining_requests[index - self._min_index] -= 1

        # Garbage collection
        while self._remaining_requests and self._remaining_requests[0] <= 0:  # The < is not a valid case, but you need it to not cause an infinite loop in case other implementation are buggy.
            self._buffer.pop(0)
            self._remaining_requests.pop(0)
            self._min_index += 1
        return result


class WirelessRecombiningObserver(ObservableObserver[Received,Tuple[Received, _Received2]]):

    def __init__(self, connection: WirelessObserverConnection[_Received2], observers: List[Observer[Tuple[Received, _Received2]]]):
        super().__init__(observers=observers)
        self._index = 0

        self._server = None
        connection._connectReceiver(self)

    def _receive(self, sample: Received, weight: float):
        self._send( (sample, self._server._request(self._index)), weight)
        self._index += 1

    def _registerServer(self, server: WirelessSplittingObserver[Any,_Received2]):
        self._server = server
