"""
One contract for all forms of start-of-word (SoW) and end-of-word (EoW).
"""
from typing import Iterable
from dataclasses import dataclass
from enum import Enum

from ..util.trie import PrefixTrie, SuffixTrie


class BoundaryMarkerLocation(str, Enum):  # The str parent allows JSON serialisation: https://stackoverflow.com/a/51976841/9352077
    ISOLATED = 1
    START = 2
    END = 3


@dataclass
class BoundaryMarker:
    substitute: str
    detached: bool
    location: BoundaryMarkerLocation

    def isolate(self, pretoken: str) -> tuple[str, str]:
        """
        Retrieve the part of a pretoken that isn't a space marker.
        """
        L = len(self.substitute)
        if self.location == BoundaryMarkerLocation.START:
            root, mark = pretoken[L:], pretoken[:L]
            if mark != self.substitute:
                root, mark = pretoken, ""
        elif self.location == BoundaryMarkerLocation.END:
            root, mark = pretoken[:len(pretoken)-L], pretoken[len(pretoken)-L:]
            if mark != self.substitute:
                root, mark = pretoken, ""
        elif self.location == BoundaryMarkerLocation.ISOLATED:
            if pretoken == self.substitute:
                root, mark = "", pretoken
            else:
                root, mark = pretoken, ""
        else:
            root, mark = pretoken, ""

        return root, mark

    def concatenate(self, root: str, marker: str) -> str:
        """
        Inverse of isolate().
        """
        if self.location == BoundaryMarkerLocation.START:
            return marker + root
        elif self.location == BoundaryMarkerLocation.END:
            return root + marker
        elif self.location == BoundaryMarkerLocation.ISOLATED:
            return root if root else marker
        else:
            return root

    def atomise(self, pretoken: str) -> tuple[str,...]:
        """
        Method for algorithms like BPE that require a word to start out as being split into characters.

        Does NOT add SoW/EoW because this is already done when you split the sentence into words.
        What it might do, however, is attach the SoW/EoW to the adjacent character.
        """
        if self.location == BoundaryMarkerLocation.START:
            chars, sow = self.isolate(pretoken)
            if not sow:
                return tuple(chars)
            else:
                if self.detached:
                    return (sow,) + tuple(chars)
                else:
                    return (sow + chars[0],) + tuple(chars[1:])

        elif self.location == BoundaryMarkerLocation.END:
            chars, eow = self.isolate(pretoken)
            if not eow:
                return tuple(chars)
            else:
                if self.detached:
                    return tuple(chars) + (eow,)
                else:
                    return tuple(chars[:-1]) + (chars[-1] + eow,)

        elif self.location == BoundaryMarkerLocation.ISOLATED:
            if pretoken == self.substitute:
                return (pretoken,)
            else:
                return tuple(pretoken)

        else:
            return (pretoken,)


# Old names for compatibility
SpaceMarker = BoundaryMarker
SpaceMarkerLocation = BoundaryMarkerLocation


def detectBoundaryMarkerFromVocabulary(vocab: Iterable[str], threshold: float=0.5) -> BoundaryMarker:
    vocab = list(vocab)
    V = len(vocab)

    trie = PrefixTrie()
    for t in vocab:
        trie.add(t)
    trie.compileRoots()

    suggested_prefix = trie
    while True:
        top = suggested_prefix.getTopChildNodes(n=1)
        if len(top) and top[0].count / V >= threshold:
            suggested_prefix = top[0]
        else:
            break

    trie = SuffixTrie()
    for t in vocab:
        trie.add(t)
    trie.compileRoots()

    suggested_suffix = trie
    while True:
        top = suggested_suffix.getTopChildNodes(n=1)
        if len(top) and top[0].count / V >= threshold:
            suggested_suffix = top[0]
        else:
            break

    found_prefix = len(suggested_prefix.root) > 0
    found_suffix = len(suggested_suffix.root) > 0
    prefix = BoundaryMarker(suggested_prefix.root, detached=suggested_prefix.root in vocab, location=BoundaryMarkerLocation.START)
    suffix = BoundaryMarker(suggested_suffix.root, detached=suggested_suffix.root in vocab, location=BoundaryMarkerLocation.END)

    if not found_prefix and not found_suffix:  # No prefix nor suffix? I guess it must be an isolated token.
        return BoundaryMarker("", detached=True, location=BoundaryMarkerLocation.ISOLATED)
    elif not found_suffix:  # No suffix? Then it's a prefix.
        return prefix
    elif not found_prefix:  # No prefix? Then it's a suffix.
        return suffix
    else:  # Prefix and suffix found? Then take the one with higher occurrence. (TODO: Alternatively, take the one with higher length.)
        if suggested_prefix.count > suggested_suffix.count:
            return prefix
        else:
            return suffix
