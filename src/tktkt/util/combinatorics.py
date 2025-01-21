"""
Mathematical computations that have to do with the combinatorics of string segmentation.
No connection with any tokeniser.
"""
from typing import List, Tuple, Dict, Sequence, Generator, Iterable, Iterator
from itertools import permutations, combinations
from functools import reduce
from collections import Counter

from math import factorial, comb
from operator import mul

from .iterables import keepFirst, T

# There are four canonical ways to represent the segmentation of a known string:
#     - A list of token strings;
#     - A list of their lengths;
#     - A list of their beginning indices;
#     - A list where each element is an inter-character position, where True is a split and False is not.
#       This can also be represented compactly by reading it as a binary number and storing the decimal representation
#       (e.g. [True,False,False,False,True,False,True] is mask 69 for 8-character strings).
Tokens = Sequence[str]
TokenLengths = Sequence[int]
TokenStartIndices = Sequence[int]
SplitMask = Sequence[bool]


def permutations_no_repeat(sequence: Sequence[T]) -> Generator[Sequence[T],None,None]:
    """
    O(n!) way of generating multiset permutations, i.e. the unique permutations of a sequence that can contain duplicate elements,
    because we consider duplicate elements to be identical and hence swapping them is meaningless.
    """
    yield from keepFirst(permutations(sequence))


def compositions(n: int) -> Generator[TokenLengths,None,None]:
    """
    Generates the compositions of the number n, i.e. the sequences (not sets!) of strictly positive integers that sum to n.
    https://en.wikipedia.org/wiki/Composition_(combinatorics)

    Related to tokenisation because these are all the sequences of token lengths a non-degenerate tokeniser can produce
    for a word of n characters. In other words: this function generates all segmentations of a string of n characters.
    The output hence contains 2^{n-1} tuples, equal to the amount of binary variations of whether there is a split on
    the n-1 inter-character positions.
    """
    for n_splits in range(n):  # 0 ... n-1 splits
        yield from compositions_k(n, k=n_splits)


def compositions_k(n: int, k: int) -> Generator[TokenLengths,None,None]:
    """
    Compositions (see above), but limited to sequences of exactly k integers.
    Has binom(n-1,k-1) total outputs (the amount of ways to choose k-1 token dividers out of n-1 inter-character positions).
    """
    for indices in combinations(range(1,n), r=k):
        indices = (0,) + indices + (n,)
        yield tuple([b - a for a, b in zip(indices[:-1], indices[1:])])


def integerPartitions_k_highMemory(n: int, k: int, upper_limit: int=None, memoisation: Dict[Tuple[int,int],List[TokenLengths]]=None) -> List[TokenLengths]:
    """Deprecated high-memory implementation of integerPartitions_k that constructs all results in memory
       simultaneously, and sorts the partitions right-to-left, both unlike integerPartitions_k."""
    assert not(n < 0 or k < 0 or (n == 0 and k != 0) or (n != 0 and k == 0))

    if memoisation is None:
        memoisation = dict()
    elif (n,k) in memoisation:
        return memoisation[(n,k)]

    if k == 0:
        return [()]

    if upper_limit is None:
        upper_limit = n-k+1  # Say you need to reach n=5 in k=3 steps. Then you can't take a step of length 4. At most, you can use one step of length 3, followed by steps of 1 after.
    upper_limit = min(n-k+1, upper_limit)
    lower_limit = 1 + (n-1) // k  # Let's say you have n = 7 to cross in 3 steps. You can't output 2, because then even if you only choose 2 from then on, you will not reach n == 0 in time.

    results = []
    for step in range(upper_limit, lower_limit-1, -1):  # Largest step first.
        paths = integerPartitions_k(n-step, k-1, upper_limit=step)
        assert len(paths) > 0  # The goal of the algorithm is to never do redundant operations, so a function call must always produce at least one result.
        for path in paths:
            results.append(path + (step,))

    memoisation[(n,k)] = results
    return results


def integerPartitions_k(n: int, k: int, prefix: TokenLengths=()) -> Iterator[TokenLengths]:
    """
    Generates only the integer compositions of length k summing to n that are in ascending order, ignoring all permutations
    of these. Importantly, this is **not** done by generating all binom(n-1)(k-1) compositions for better time complexity.
    https://en.wikipedia.org/wiki/Integer_partition
    https://math.stackexchange.com/a/4967437/615621

    Less relevant in tokenisation because permutations of token lengths do not give the same segmentation. However, if
    you need to subdivide the set of segmentations (and hence compositions) of a string by the permutation equivalence
    class of the token lengths, this function will give you one representative result for each class.

    :param n: total that the sequences must sum to.
    :param k: amount of non-zero positive integers in the sequences.
    """
    assert not(n < 0 or k < 0 or (n == 0 and k != 0) or (n != 0 and k == 0))

    if k == 0:
        yield prefix
        return  # The 'None' output by this return does not make its way to

    lower_limit = n if k == 1 else prefix[-1] if prefix else 1  # Lower limit is either 1 (no history) or the last number used. When you only have one step left, the lower limit is not the last number used.
    upper_limit = n // k                                        # Upper limit: you can't take a step so big that it exceeds n if repeated. Let's say you have n = 7 to cross in 3 steps. You can then output 2 at most, because if you output 3, you'll end up at 3*k == 9 > n. You can only start outputting step=3 at k=3 from n>=9 onward.

    for step in range(lower_limit,upper_limit+1):
        yield from integerPartitions_k(n-step, k-1, prefix=prefix + (step,))


def permutationToIdentifier(permutation: Sequence) -> int:
    """
    Produces an integer which is unique among all the permutations of the given sequence.
    Works even if it does not consist of the integers {0,...,n-1} and even if it contains duplicates (which can be
    permuted without changing the result because they are equivalent).
    https://stackoverflow.com/a/78953283/9352077
    """
    counter = Counter(permutation)
    ordered_elements = sorted(counter)

    n = len(permutation)
    result = 0
    for i, element in enumerate(permutation):
        for j in range(ordered_elements.index(element)):
            lower_element = ordered_elements[j]
            if counter[lower_element] > 0:
                # "How many solutions could we have formed if we had chosen j instead of the given element at this position?"
                counter[lower_element] -= 1
                result += countMultisetPermutations(n-i-1, counter.values())
                counter[lower_element] += 1

        counter[element] -= 1

    return result


def countMultisetPermutations(n: int, multiplicities: Iterable[int]) -> int:
    """
    A sequence of n distinct items has n! permutations. A sequence of n items in general has n! permutations divided by
    all possible permutations of duplicate items within those permutations, because those permutations are all equivalent.
    https://en.wikipedia.org/wiki/Permutation#Permutations_of_multisets

    In other words: a generalised form of permutations.
    """
    return factorial(n) // reduce(mul, map(lambda x: factorial(x), multiplicities))


def countMultisetPermutationsOfSequence(sequence: Sequence):
    return countMultisetPermutations(len(sequence), Counter(sequence).values())


def countCompositions_k(n: int, k: int):
    """
    For tokenisation, returns the amount of ways in which a string of n characters can be segmented into k tokens
    (which are always non-empty).

    :param n: the amount of characters.
    :param k: the exact amount of tokens they should be segmented into.
    """
    return comb(n-1, k-1)


countSegmentationsOfKTokens = countCompositions_k
equivalenceClassesForSegmentationsOfKTokens = integerPartitions_k
equivalenceClassSize = countMultisetPermutationsOfSequence


def getLOCKey(token_lengths: TokenLengths) -> int:
    """
    Applies my (as far as I know) "length-ordered composition (LOC)" function, which is a bijective mapping between the
    segmentations of an n-character string (i.e. compositions of the number n) and the integers {0, ..., 2^{n-1} - 1}.
    In other words, assigns a unique key to every segmentation of a string with no unused keys between the lowest and highest key,
    while applying a special ordering.

    Segmentations of an n-character string can be ordered by converting the segmentation mask from binary to decimal.
    However, there is basically no relationship to the token lengths in this ordering (since e.g. 01111 and 10000 are neighbours).

    This function gives the order key for a different ordering:
        1. first order by amount of tokens;
        2. within the same token amount: partition the segmentations based on the result of sorting their token lengths,
           and sort those subsets by tuple comparison of that result.
        3. within the same subset: don't sort (because they're permutations of the same sequence of token lengths, and
           we don't care about their order here), although have a deterministic order.

    The key is computed explicitly, without needing to know any other key. Hence, this function's output range is
    exponential in the length of its input, without exponential time or space complexity (which you'd get if you
    computed the key by first generating all O(2^{n-1}) segmentations in space and time, then doing an O((n-1) 2^{n-1}) sort,
    and finally doing an O(2^{n-1}) seek).
    """
    n = sum(token_lengths)
    k = len(token_lengths)

    # + The amount of segmentations with fewer tokens than this one.
    n_shorter_segmentations = 0
    for smaller_k in range(1,k):
        n_shorter_segmentations += countSegmentationsOfKTokens(n, smaller_k)

    # + Sizes of all preceding permutation equivalence classes for (n,k), in order.
    equivalence_classes = equivalenceClassesForSegmentationsOfKTokens(n,k)
    this_class = tuple(sorted(token_lengths))
    n_segmentations_equalength_lowerclass = 0
    for c in equivalence_classes:
        if c == this_class:
            break
        n_segmentations_equalength_lowerclass += equivalenceClassSize(c)

    # + Its permutation number inside its own permutation equivalence class.
    n_segmentations_equalength_equiclass_lowerperm = permutationToIdentifier(token_lengths)
    return n_shorter_segmentations + n_segmentations_equalength_lowerclass + n_segmentations_equalength_equiclass_lowerperm
