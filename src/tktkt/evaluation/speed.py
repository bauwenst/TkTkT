from typing import Iterable, Tuple

from ..interfaces.tokenisers import Tokeniser
from ..util.timing import Timer


def secondsPerTokenisation(tokeniser: Tokeniser, texts: Iterable[str]) -> Tuple[float,float]:
    """
    Returns the average and standard deviation of the amount of seconds it takes to preprocess and then tokenise the
    given texts. (Note: these numbers are affected by sentence length.)
    """
    t = Timer()
    t.start()
    for text in texts:
        tokeniser.prepareAndTokenise(text)
        t.lap()
    return t.averageLapTime(), t.stdLapTime()
