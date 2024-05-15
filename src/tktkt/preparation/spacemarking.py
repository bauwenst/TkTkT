"""
One contract for all forms of start-of-word (SoW) and end-of-word (EoW).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class SpaceMarkerLocation(Enum):
    ISOLATED = 1
    START = 2
    END = 3


@dataclass
class SpaceMarker:
    substitute: str
    detached: bool
    location: SpaceMarkerLocation

    def isolate(self, pretoken: str) -> Tuple[str, str]:
        """
        Retrieve the part of a pretoken that isn't a space marker.
        """
        L = len(self.substitute)
        if self.location == SpaceMarkerLocation.START:
            root, mark = pretoken[L:], pretoken[:L]
            if mark != self.substitute:
                root, mark = pretoken, ""
        elif self.location == SpaceMarkerLocation.END:
            root, mark = pretoken[:len(pretoken)-L], pretoken[len(pretoken)-L:]
            if mark != self.substitute:
                root, mark = pretoken, ""
        elif self.location == SpaceMarkerLocation.ISOLATED:
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

        FIXME: Possibly doesn't work properly if the pretoken does not contain the self.marker.
        """
        if self.location == SpaceMarkerLocation.START:
            chars, sow = self.isolate(pretoken)
            if self.detached:
                return [sow] + list(chars)
            else:
                return [sow + chars[0]] + list(chars[1:])
        elif self.location == SpaceMarkerLocation.END:
            chars, eow = self.isolate(pretoken)
            if self.detached:
                return list(chars) + [eow]
            else:
                return list(chars[:-1]) + [chars[-1] + eow]
        elif self.location == SpaceMarkerLocation.ISOLATED:
            if pretoken == self.substitute:
                return [pretoken]
            else:
                return list(pretoken)
        else:
            return [pretoken]
