from typing import Iterable
from enum import Enum
import numpy as np

import hashlib as _lib

from .iterables import intercalate, cumsum


def shash(s: str, bits: int=40) -> str:
    """
    Stable, short, SHA string (SSSS) hash. Hexadecimal representation of the bitstring digest from the chosen hash.
    All hashing in TkTkT must use this function. The built-in hash() is only deterministic within the same runtime,
    whilst this function is stable across runtimes.

    :param bits: Amount of bits the output represents, up to 256. The length of the output will be this number
                 divided by 4 (since 4 bits make 1 hex). You can expect a collision after 2^{bits/2} applications
                 of this function. So, the default of 40 bits is safe for anything where you expect fewer than 1_000_000
                 tries to collide.
    """
    assert 0 < bits <= 256 and bits % 4 == 0
    return _lib.sha256(s.encode("utf-8")).hexdigest()[:bits//4]


def prefixIfNotEmpty(prefix: str, s: str) -> str:
    return prefix*bool(s) + s


def suffixIfNotEmpty(s: str, suffix: str) -> str:
    return s + suffix*bool(s)


def circumfixIfNotEmpty(prefix: str, s: str, suffix: str) -> str:
    return prefix*bool(s) + s + suffix*bool(s)


def interfixIfNotEmpty(s1: str, interfix: str, s2: str) -> str:
    return s1 + interfix*(bool(s1) and bool(s2)) + s2


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


def getAlphabet(strings: Iterable[str]) -> set[str]:
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


def surround(text: str, frame_character: str= "#", frame_width: int=1) -> str:
    assert len(frame_character) == 1
    lines = text.split("\n")
    base_width = max(map(len, lines))

    full_width = base_width + 2 + 2*frame_width
    output_lines = []
    output_lines.append(frame_character*full_width)
    for line in lines:
        output_lines.append(frame_character*frame_width + " " + line + " "*(base_width - len(line)) + " " + frame_character*frame_width)
    output_lines.append(frame_character*full_width)
    return "\n".join(output_lines)

########################################################################################################################


from .types import Tokens, TokenStartIndices, SplitMask

def indicesToTokens(text: str, starts_of_tokens: TokenStartIndices) -> Tokens:
    return [text[start_idx:end_idx] for start_idx, end_idx in zip(starts_of_tokens, starts_of_tokens[1:] + [len(text)])]

def maskToTokens(text: str, mask: SplitMask) -> Tokens:
    return indicesToTokens(text, [0] + (np.nonzero(mask)[0] + 1).tolist())

def bitstringToTokens(text: str, bitmap: str) -> Tokens:
    return indicesToTokens(text, starts_of_tokens=[0] + [i+1 for i,c in enumerate(bitmap) if c == "1"])


def tokensToIndices(tokens: Tokens) -> TokenStartIndices:
    return [0] + list(cumsum(map(len, tokens)))[:-1]

def tokensToMask(tokens: Tokens) -> SplitMask:
    return sum(intercalate(map(lambda i: (i-1)*[0], map(len, tokens)), [1]), start=[])
    # return list(map(int, tokensToBitstring(tokens)))

def tokensToBitstring(tokens: Tokens) -> str:
    return "".join(intercalate(map(lambda i: (i-1)*"0", map(len, tokens)), "1"))
    # return "".join(map(str, tokensToMask(tokens)))
