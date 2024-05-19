"""
One contract for all forms of start-of-word (SoW) and end-of-word (EoW).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class BoundaryMarkerLocation(Enum):
    ISOLATED = 1
    START = 2
    END = 3


@dataclass
class BoundaryMarker:
    substitute: str
    detached: bool
    location: BoundaryMarkerLocation

    def isolate(self, pretoken: str) -> Tuple[str, str]:
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

    def intoCharacters(self, pretoken: str):
        """
        Method for algorithms like BPE that require a word to start out as being split into characters.

        Does NOT add SoW/EoW because this is already done when you split the sentence into words.
        What it might do, however, is attach the SoW/EoW to the adjacent character.
        """
        if self.location == BoundaryMarkerLocation.START:
            chars, sow = self.isolate(pretoken)
            if not sow:
                return list(chars)
            else:
                if self.detached:
                    return [sow] + list(chars)
                else:
                    return [sow + chars[0]] + list(chars[1:])
        elif self.location == BoundaryMarkerLocation.END:
            chars, eow = self.isolate(pretoken)
            if not eow:
                return list(chars)
            else:
                if self.detached:
                    return list(chars) + [eow]
                else:
                    return list(chars[:-1]) + [chars[-1] + eow]
        elif self.location == BoundaryMarkerLocation.ISOLATED:
            if pretoken == self.substitute:
                return [pretoken]
            else:
                return list(pretoken)
        else:
            return [pretoken]


# Old names for compatibility
SpaceMarker = BoundaryMarker
SpaceMarkerLocation = BoundaryMarkerLocation