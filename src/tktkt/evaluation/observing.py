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
from typing import List, TypeVar, Generic, Iterator, Any, Callable, Tuple, Union

from ..interfaces import Preprocessor
from ..interfaces.tokenisers import Tokeniser
from ..interfaces.observers import Observer, ObservableObserver, ImmediatelyObservableObserver, Observable, ObservableMeta, ObserverEarlyExit, FinallyObservableObserver, ObservableRoot, Received, Sent
from ..util.dicts import optionalDataclassToDict
from ..util.iterables import dunion
from ..util.types import NamedIterable, Tokens, HoldoutState


class FutureObserver(Observer[Received]):
    """
    Gives access to the most recently received value.
    """
    def __init__(self):
        self._value = None

    def _initialise(self, global_run_identifier: str):
        self._value = None

    def _receive(self, sample: Received, _):
        self._value = sample

    def _finish(self):
        pass

    def resolve(self) -> Received:
        if self._value is None:
            raise RuntimeError("Attempted to resolve Future before the run that sets its value (or no such run exists).")
        return self._value


class AppendToListObserver(Observer[Received]):
    """
    Gives access to the entire history of received values as one flat chronological list.
    """
    def __init__(self, list_to_append_to: list):
        self._list_reference = list_to_append_to

    def _initialise(self, global_run_identifier: str):
        pass

    def _receive(self, sample: Received, _):
        self._list_reference.append(sample)

    def _finish(self):
        pass

    def listcopy(self) -> list[Received]:
        return list(self._list_reference)


class DataclassObserver(Observer[Any]):
    """
    Observer meant to be put at various points in the hierarchy, to collect dictionaries/dataclasses that are
    supposed to be saved together as one row in a CSV file.

    The user decides when a new row is started by calling .fence() on the observer.
    """

    def __init__(self, fence_on_assemble: bool=True, field_suffix: str=""):
        self._current_list    = []
        self._completed_lists = []

        self._fence_on_assemble = fence_on_assemble
        self._suffix = field_suffix
        self._is_proxy_for: DataclassObserver = None

    def withSuffix(self, suffix: str) -> "DataclassObserver":
        """
        Get an object which still funnels received dataclasses to the current object, except with a suffix added to
        the end of the fields of the received dataclasses.
        """
        new_observer = DataclassObserver(fence_on_assemble=self._fence_on_assemble, field_suffix=suffix)
        new_observer._makeProxyFor(self)
        return new_observer

    def _makeProxyFor(self, destination: "DataclassObserver"):
        if self._current_list or self._completed_lists:
            raise RuntimeError("Cannot make proxy out of observer that has seen data already.")
        self._is_proxy_for = destination

    def addMetadata(self, metadata: dict):
        """Add metadata to the current row."""
        self._receive(metadata, 1)

    def fence(self):
        """Finish the current collection and start a new collection."""
        if self._is_proxy_for is not None:
            raise RuntimeError(f"Cannot fence proxy. You can only fence the original observer, {self._is_proxy_for}.")
        self._completed_lists.append(self._current_list)
        self._current_list = []

    def assemble(self) -> List[dict]:
        """For each collection that has been fenced, pool together all the dataclasses/dictionaries."""
        if self._is_proxy_for is not None:
            raise RuntimeError(f"Cannot assemble proxy. You can only assemble the original observer, {self._is_proxy_for}.")
        if self._fence_on_assemble and self._current_list:
            self.fence()
        return [dunion(map(optionalDataclassToDict, dicts)) for dicts in self._completed_lists]

    def _initialise(self, global_run_identifier: str):
        pass

    def _receive(self, sample: Any, _):
        if self._suffix:
            sample = {f"{k}_{self._suffix}": v for k, v in optionalDataclassToDict(sample).items()}

        if self._is_proxy_for is None:
            self._current_list.append(sample)
        else:
            self._is_proxy_for._receive(sample, _)

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


class ObservablePreprocessor(ObservableObserver[str,str]):

    def __init__(self, preprocessor: Preprocessor, observers: List[Observer[str]]):
        super().__init__(observers=observers)
        self.preprocessor = preprocessor

    def _receive(self, sample: str, weight: float):
        for pretoken in self.preprocessor.do(sample):
            self._send(pretoken, weight)


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
