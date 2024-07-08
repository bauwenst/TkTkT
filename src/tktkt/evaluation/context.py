"""
Evaluation of the context around tokens.
"""
from typing import Iterable, Dict, Set
from dataclasses import dataclass
from collections import defaultdict, Counter

from ..interfaces.tokeniser import Tokeniser


@dataclass
class AV:
    left: int
    right: int
    both: int


def getAccessorVarieties(tokeniser: Tokeniser, word_iterator: Counter, do_count_ends: bool=True) -> Dict[str, AV]:
    # Everything you have seen to the left and right of a given type
    left_adjacents: Dict[str, Set[str]]  = defaultdict(set)
    right_adjacents: Dict[str, Set[str]] = defaultdict(set)

    # When a type appears at the start of a word, technically it still has leftward variety, namely whatever token the
    # previous word ended with. However, which token that is exactly matters much less because there is no connection
    # between the characters of the previous word and of the current word (only the meanings). Hence, you assume that
    # it is entirely random and hence you are guaranteed an accessor every time you are at the start. Same for the end.
    left_pseudos: Dict[str, int]  = defaultdict(int)
    right_pseudos: Dict[str, int] = defaultdict(int)

    for word, frequency in word_iterator.items():
        tokens = tokeniser.prepareAndTokenise(word)

        # Edge tokens
        left_pseudos[tokens[0]] += frequency
        right_pseudos[tokens[-1]] += frequency

        if len(tokens) > 1:
            right_adjacents[tokens[0]].add(tokens[1])
            left_adjacents[tokens[-1]].add(tokens[-2])

        # Middle tokens
        for i in range(1,len(tokens)-1):
            center = tokens[i]
            left_adjacents[center].add(tokens[i-1])
            right_adjacents[center].add(tokens[i+1])

    all_types = set(left_pseudos) | set(right_pseudos) | set(left_adjacents) | set(right_adjacents)
    return {
        t: AV(
            left=len(left_adjacents[t])                      + do_count_ends*left_pseudos[t],
            right=len(right_adjacents[t])                    + do_count_ends*right_pseudos[t],
            both=len(left_adjacents[t] | right_adjacents[t]) + do_count_ends*(left_pseudos[t] + right_pseudos[t])
        )
        for t in all_types
    }
