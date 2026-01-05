from typing import List, Iterable, Set
from enum import Enum
import numpy as np

from .iterables import intercalate, cumsum


def prefixIfNotEmpty(prefix: str, s: str):
    return prefix*bool(s) + s


def suffixIfNotEmpty(s: str, suffix: str):
    return s + suffix*bool(s)


def underscoreIfNotEmpty(s: str, prefix_not_suffix: bool=True) -> str:
    if s:
        return "_" + s if prefix_not_suffix else s + "_"
    else:
        return s


def findLongestCommonPrefix(strings: Iterable[str]) -> str:
    ref = None
    for string in strings:
        # Trivial cases
        if ref is None:
            ref = string
            continue
        elif not ref:
            break

        # Shorten the reference
        if len(string) < len(ref):
            ref = ref[:len(string)]
        for i in range(len(ref)):
            if string[i] != ref[i]:
                ref = ref[:i]
                break

    return ref or ""


def anySubstringIn(substrings: Iterable[str], string: str) -> bool:
    return any(sub in string for sub in substrings)


def getAlphabet(strings: Iterable[str]) -> Set[str]:
    alphabet = set()
    for s in strings:
        alphabet.update(s)
    return alphabet


def indent(level: int, multiline_string: str, tab: str=" "*4) -> str:
    """
    Prefix each line in the given string by the given tab (except the last line if it is empty).
    """
    # As a one-liner: "".join(tab + line + "\n" for line in s.splitlines())
    lines = multiline_string.split("\n")
    if lines[-1] == "":
        lines.pop()
        add_empty_line = True
    else:
        add_empty_line = False

    lines = [tab*level + line for line in lines]
    return "\n".join(lines) + "\n"*add_empty_line


def alignCharacter(multiline_string: str, character_to_align: str) -> str:
    """
    Add spaces right in front of the first occurrence of the given character for each line in the string, such that
    the character aligns. Useful for aligning equals signs in subsequent assignments.
    """
    lines = multiline_string.split("\n")
    character_locations = [line.find(character_to_align) for line in lines]
    move_character_to = max(character_locations)
    for i, (line, loc) in enumerate(zip(lines, character_locations)):
        if loc < 0:
            continue

        lines[i] = line[:loc] + " "*(move_character_to - loc) + line[loc:]

    return "\n".join(lines)


class Case(Enum):
    SNAKE  = 0  # snake_case
    KEBAB  = 1  # kebab-case
    PASCAL = 2  # PascalCase
    CAMEL  = 3  # camelCase
    FLAT   = 4  # flatcase
    SPACED = 5  # spaced case


def convertCase(text: str, from_case: Case, to_case: Case) -> str:
    if from_case == to_case:
        return text
    if not text:
        return text

    if   from_case == Case.SNAKE:
        parts = text.split("_")
    elif from_case == Case.KEBAB:
        parts = text.split("-")
    elif from_case == Case.SPACED:
        parts = text.split(" ")
    elif from_case == Case.PASCAL or from_case == Case.CAMEL:
        indices = []
        for i,c in enumerate(text):
            if c.isupper():
                indices.append(i)
        if not indices or indices[0] != 0:
            indices.insert(0,0)
        indices.append(len(text))
        parts = [text[i:j] for i,j in zip(indices[:-1], indices[1:])]
    elif from_case == Case.FLAT:
        raise ValueError("Cannot convert from flatcase to other cases")
    else:
        raise NotImplementedError(from_case)

    if   to_case == Case.SNAKE:
        return "_".join(parts)
    elif to_case == Case.KEBAB:
        return "-".join(parts)
    elif to_case == Case.SPACED:
        return " ".join(parts)
    elif to_case == Case.FLAT:
        return "".join(parts)
    elif to_case == Case.PASCAL:
        return "".join(p.capitalize() for p in parts)
    elif to_case == Case.CAMEL:
        result = "".join(p.capitalize() for p in parts)
        return result[0].lower() + result[1:]
    else:
        raise NotImplementedError(to_case)


########################################################################################################################


def indicesToTokens(text: str, starts_of_tokens: List[int]) -> List[str]:
    return [text[start_idx:end_idx] for start_idx, end_idx in zip(starts_of_tokens, starts_of_tokens[1:] + [len(text)])]

def maskToTokens(text: str, mask: List[int]) -> List[str]:
    return indicesToTokens(text, [0] + (np.nonzero(mask)[0] + 1).tolist())

def bitstringToTokens(text: str, bitmap: str) -> List[str]:
    return indicesToTokens(text, starts_of_tokens=[0] + [i+1 for i,c in enumerate(bitmap) if c == "1"])


def tokensToIndices(tokens: List[str]) -> List[int]:
    return [0] + list(cumsum(map(len, tokens)))[:-1]

def tokensToMask(tokens: List[str]) -> List[int]:
    return sum(intercalate(map(lambda i: (i-1)*[0], map(len, tokens)), [1]), start=[])
    # return list(map(int, tokensToBitstring(tokens)))

def tokensToBitstring(tokens: List[str]) -> str:
    return "".join(intercalate(map(lambda i: (i-1)*"0", map(len, tokens)), "1"))
    # return "".join(map(str, tokensToMask(tokens)))
