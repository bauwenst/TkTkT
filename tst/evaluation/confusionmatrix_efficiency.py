from src.tktkt.util.timing import timeit
from src.tktkt.evaluation.morphological import compareSplits, compareSplits2, compareSplits3

from bpe_knockout.project.config import morphologyGenerator


examples = [("ab c d efghi j", "a bcd ef ghij"),
            ("abc def", "a bcd e f")]


def correctness():
    for candidate, reference in examples:
        print(compareSplits(candidate, reference))
        print(compareSplits2(candidate, reference))
        print(compareSplits3(candidate, reference))
        print()


@timeit
def speed(variant: int):
    if variant == 1:
        method = compareSplits
    elif variant == 2:
        method = compareSplits2
    elif variant == 3:
        method = compareSplits3
    else:
        return

    for candidate, reference in examples*500_000:
        method(candidate, reference)


if __name__ == "__main__":
    correctness()  # Also does numba compilation if you need it.

    speed(1)
    speed(2)
    speed(3)