"""
Abstract classes to do with the observable-observer framework.
You should import from this file only when you are developing new observers/observables, otherwise you probably need
the tktkt.evaluation.observing submodule.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterator, Any, Callable
from pathlib import Path

import numpy.random as npr
import traceback
import colorama

from ..paths import TkTkTPaths
from ..util.interfaces import C
from ..util.strings import indent, suffixIfNotEmpty, prefixIfNotEmpty
from ..util.interfaces import Cache, Cacheable


Received = TypeVar("Received")
Sent     = TypeVar("Sent")


def _formatException(exception_string: str) -> str:
    return indent(1, f"{colorama.Fore.YELLOW}{exception_string}{colorama.Fore.RESET}")


def _formatEarlyExit(msg: str) -> str:
    return indent(1, f"{colorama.Fore.CYAN}{msg}{colorama.Fore.RESET}")


def _formatExit(msg: str) -> str:
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

    def __init__(self, observers: list[Observer[Sent]]):
        assert observers, "An observable must have at least one observer."
        self.observers = observers
        self.done = []
        self.dead = []

    @abstractmethod
    def _nodeIdentifier(self) -> str:
        """Disambiguates instances of this observable. Observables with the same identifier path are supposed to produce the same output every time."""
        pass

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
                print(_formatEarlyExit(f"Observer finished early: {observer.__class__.__name__}"))
                done.append(observer)
            except:
                print(_formatException(f"Observer failed: {observer.__class__.__name__}"))
                print(_formatException(traceback.format_exc()))
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

    def _initialiseObservers(self, identifier_so_far: str):  # This method only to be called by users of this class, not by the class itself.
        self._callObservers(lambda observer: observer._initialise(identifier_so_far + prefixIfNotEmpty("_", self._nodeIdentifier().replace("_", "-"))))

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
                print(_formatException(traceback.format_exc()))

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

    def __init__(self, cache_disambiguator: str, observers: list[Observer[Sent]]):
        """
        :param cache_disambiguator: disambiguates runs whose root instances are otherwise identical.
                                    TODO: A smarter way to do this would be to instead have nodes in the tree
                                          add on to the name of the root. Because indeed, the only thing that CAN
                                          change in the experiment if the root node is the same but the cache is not,
                                          is any node between the two, e.g. the tokeniser, which the user is now asked to identify in the disambiguator manually.
        """
        super().__init__(observers=observers)
        self._cache_disambiguator = cache_disambiguator

    @abstractmethod
    def _stream(self) -> Iterator[tuple[Sent,float]]:
        pass

    def _globalRunIdentifier(self) -> str:
        """Informs observers which of their caches to use."""
        return suffixIfNotEmpty(self._cache_disambiguator, "_") + self._nodeIdentifier()

    def run(self):
        print(_formatExit(f"Running {self._globalRunIdentifier()}..."))

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

        print(_formatExit("ObservableRoot finished running."))
        print(_formatExit(self.observerStatusReport()))


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


class ImmediatelyObservableObserver(ObservableObserver[Received, Sent]):
    """
    Special kind of observable observer which outputs exactly one sample whenever it receives a sample.
    """

    @abstractmethod
    def _transit(self, sample: Received, weight: float) -> Sent:
        pass

    def _receive(self, sample: Sent, weight: float):
        self._send(self._transit(sample, weight), weight)


CacheableSent = TypeVar("CacheableSent", bound=Cacheable)

class FinallyObservableObserver(ObservableObserver[Received, CacheableSent], Cache[CacheableSent]):
    """
    Special kind of ObservableObserver which only outputs something when its Observable is FINISHED.
    Its observers need to be sent the result before they are finished too.

    This is the one type of Observer that has caching.
    """

    def __init__(self, cache_disambiguator: str= "", disable_cache: bool=False, observers: list[Observer[CacheableSent]]=None):
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
        if not self._disable_cache and self._cacheType().exists(self._cachePath(self._cacheIdentifier())):
            raise ObserverEarlyExit()

    @abstractmethod
    def _compute(self) -> CacheableSent:
        pass

    @abstractmethod
    def _cacheSubfolders(self) -> list[str]:
        pass

    def _cacheIdentifier(self) -> str:  # In Vocabularisers, you don't need to use fields for this function, since the identifier is known in the same context that ._cacheRun is run. Not only is that not the case here, but also, the possibility of name aliasing exists, which is never the case for Vocabularisers.
        return self._stored_global_run_identifier + prefixIfNotEmpty("_", self._disambiguation_identifier)

    def _cachePath(self, unambiguous_cache_identifier: str) -> Path:  # TODO: There is a case to be made that actually, the identifier should be FIRST, so that all results of one run are grouped in their own little file system.
        return TkTkTPaths.pathToEvaluations(*self._cacheSubfolders(), unambiguous_cache_identifier)

    def _cacheFinalise(self, loaded: C) -> C:
        return loaded

    def _finishAsObserver(self):
        result = self._cacheRun(self._cacheIdentifier(), self._compute)
        self._send(result, 1)
        # Only after THIS will its own observers be finished.
