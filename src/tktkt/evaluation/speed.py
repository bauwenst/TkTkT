from typing import Iterable

from ..interfaces.tokeniser import Tokeniser
from ..util.timing import Timer


def secondsPerTokenisation(tokeniser: Tokeniser, texts: Iterable[str]) -> float:
    t = Timer()
    t.start()
    for text in texts:
        tokeniser.prepareAndTokenise(text)
        t.lap()
    return t.averageLapTime()
