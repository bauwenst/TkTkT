"""
Parallelisation functions.
"""
from typing import TypeVar, Callable, ParamSpec, Sequence

import concurrent.futures
from tqdm.auto import tqdm

K = TypeVar("K")
P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["batchItems", "batchElements"]


def identifiedCall(id: K, f: Callable[P, R], *args: P, **kwargs: P) -> tuple[K, R]:
    return id, f(*args, **kwargs)


def batchItems(callables: dict[K, tuple[Callable[P, R], tuple[P]]], n_parallel_processes: int, message: str="Batch computation") -> dict[K, R]:  # Without the tuple[ ] around P, it doesn't work. Probably because mypy has a bug such that when a ParamSpec appears in the outer tuple[ ], the entire tuple, including the arguments that come before, are demanded to be of type P.
    """
    Runs a set of callables with the same signature by dispatching them to parallel processes, and links the results to
    each callable's identification key so that the parallelisation doesn't lose this ordering.

    :param callables: A set of identified functions and arguments with identical signature. For example: let's say we
                      have functions f1, f2, f3 associated respectively with the languages English, French, and Dutch,
                      all with signatures (str, float, list[int]) -> int. You could call this procedure as follows:

                            batchItems({
                                langcodes.Language("English"): (f1, ("dog",   6.9,    [1,2,3])),
                                langcodes.Language("French"):  (f2, ("chien", 4.20,   [])),
                                langcodes.Language("Dutch"):   (f3, ("hond",  800.85, [6,7])),
                            })

    :return: A dictionary equivalent to {id: f(*args) for id, (f, args) in callables.items()}.
    """
    if n_parallel_processes > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_parallel_processes) as executor:
            futures = [executor.submit(identifiedCall, id, f, *args) for id, (f, args) in callables.items()]
            results = [yielded_result.result() for yielded_result in (concurrent.futures.as_completed(futures) if not message else tqdm(concurrent.futures.as_completed(futures), desc=message, total=len(futures)))]
        return dict(results)
    else:
        return {id: f(*args) for id, (f, args) in callables.items()}


def batchElements(callables: Sequence[tuple[Callable[P, R], tuple[P]]], n_parallel_processes: int) -> list[R]:
    """Same as batchItems() except using an ordered list rather than a dictionary with keys."""
    results = batchItems({i: tuples for i, tuples in enumerate(callables)}, n_parallel_processes=n_parallel_processes)
    return [results[key] for key in sorted(results.keys())]
