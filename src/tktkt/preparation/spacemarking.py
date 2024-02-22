"""
One contract for all forms of start-of-word (SoW) and end-of-word (EoW).
"""
from dataclasses import dataclass
from enum import Enum


class SpaceMarkerLocation(Enum):
    TOKEN = 1
    START = 2
    END   = 3


@dataclass
class SpaceMarker:
    substitute: str
    detached: bool
    location: SpaceMarkerLocation


ROBERTA_SPACING = SpaceMarker("Ġ", location=SpaceMarkerLocation.START, detached=True)
