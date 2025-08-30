"""
The Observable/Observer system was designed to solve two related problems:
    (1) Having to iterate over the same corpus many times when computing different metrics, often in multiple runs;
    (2) Having to do the work of tokenisation again every time when iterating over the corpus.

The idea is that you iterate over a corpus once. If any metric needs tokens, you tokenise exactly once, and then you
distribute your tokens to all the metrics that need them.

It's kind of a pub/sub or callback or recursive map() system.

TODO: We currently don't have support for frequency.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, Generic, Iterator, Any

import traceback

from ..interfaces import Preprocessor
from ..interfaces.tokeniser import Tokeniser
from ..util.types import NamedIterable, Tokens

Received = TypeVar("Received")
Sent     = TypeVar("Sent")


class Observer(ABC, Generic[Received]):
    """
    Receives data and processes it.

    Note 1: Is not aware of who gives it data. That does not just mean "doesn't know which Observable gives it data",
            but also "doesn't even know an Observable gives it data". Thus, making somesample an Observer does not mean
            that it always needs to see the data it receives in the context of an Observable's stream.

    Note 2: By default, does not send its results to other Observers (which is fine for e.g. numerical metrics).
    """

    @abstractmethod
    def _initialise(self, identifier: str):
        pass

    @abstractmethod
    def _receive(self, sample: Received):
        """Should raise an exception when this observer should be disabled."""
        pass

    @abstractmethod
    def _finish(self):
        pass


class PrintingObserver(Observer[Any]):

    def _initialise(self, identifier: str):
        pass

    def _receive(self, sample: Any):
        print(sample)

    def _finish(self):
        pass


class Observable(ABC, Generic[Sent]):
    """
    TODO: There should be some mechanism that puts observers in a separate inactive queue when
          (1) they indicate that they have already been run or
          (2) when they error during execution.
        Basically, observers are disposable limbs of the observable.
    """

    def __init__(self, observers: List[Observer[Sent]]):
        assert observers
        self.observers = observers
        self.inactive = []

    @abstractmethod
    def _identify(self) -> str:
        """Method used to inform observers which of their caches to use."""
        pass

    def _send(self, sample: Sent):
        """Raises an exception when this observable has no more receivers."""
        failed = []

        for observer in self.observers:
            try:
                observer._receive(sample)
            except:
                print("Observer failed and moved to the inactive queue.")
                print(traceback.format_exc())
                failed.append(observer)

        for observer in failed:
            self.inactive.append(observer)
            self.observers.remove(observer)
            if not self.observers:
                raise Exception()


class ObservableRoot(Observable[Sent]):
    """Observable with a method that opens a constant stream of samples."""

    @abstractmethod
    def _stream(self) -> Iterator[Sent]:
        pass

    def run(self):
        id = self._identify()
        for observer in self.observers:
            observer._initialise(id)
        for sample in self._stream():
            try:
                self._send(sample)
            except:
                break
        for observer in self.observers:
            observer._finish()


class ObservableIterable(ObservableRoot[Sent]):
    """
    Observable that calls its send() method itself by iterating an iterable.
    Equivalently, a reverse iterable, which rather than client code iterating over it, iterates over client code.
    """

    def __init__(self, iterable: NamedIterable[Sent], observers: List[Observer[Sent]]):
        super().__init__(observers)
        self.iterable = iterable

    def _identify(self) -> str:
        return self.iterable.name

    def _stream(self):
        yield from self.iterable


class ObservableObserver(Observer[Received], Observable[Sent]):
    """
    An observer which itself is observed.
    """
    @abstractmethod
    def _initialise_self(self, identifier: str):
        pass

    @abstractmethod
    def _finish_self(self):
        pass

    def _initialise(self, identifier: str):
        for observer in self.observers:
            observer._initialise(identifier)
        self._initialise_self(identifier)

    def _finish(self):
        self._finish_self()
        for observer in self.observers:
            observer._finish()


class ImmediatelyObservableObserver(ObservableObserver[Received,Sent]):
    """
    Special kind of observable observer which outputs exactly one sample whenever it receives a sample.
    """
    
    @abstractmethod
    def _transit(self, sample: Received) -> Sent:
        pass

    def _receive(self, sample: Sent):
        self._send(self._transit(sample))


class ObservableTokeniser(ImmediatelyObservableObserver[str,Tokens]):

    def __init__(self, tokeniser: Tokeniser, observers: List[Observer[Tokens]]):
        super().__init__(observers)
        self.tokeniser = tokeniser

    def _transit(self, sample: str) -> Tokens:
        return self.tokeniser.prepareAndTokenise(sample)


class ObservablePreprocessor(ImmediatelyObservableObserver[str,Tokens]):

    def __init__(self, preprocessor: Preprocessor, observers: List[Observer[Tokens]]):
        super().__init__(observers)
        self.preprocessor = preprocessor

    def _transit(self, sample: str) -> Tokens:
        return self.preprocessor.do(sample)


class FinallyObservableObserver(ObservableObserver[Received, Sent]):
    """
    Special kind of ObservableObserver which only outputs something when its Observable is FINISHED.
    Its observers need to be sent the result before they are finished too.
    """

    @abstractmethod
    def _compute(self) -> Sent:
        pass

    @abstractmethod
    def _finish_self_after_send(self):
        pass

    def _finish_self(self):
        self._send(self._compute())
        self._finish_self_after_send()
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
