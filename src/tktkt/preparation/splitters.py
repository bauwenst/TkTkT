from typing import List
from abc import ABC, abstractmethod

from ..preparation.spacemarking import SpaceMarker, MarkerLocation


class Pretokeniser(ABC):
    @abstractmethod
    def splitSentence(self, text: str) -> List[str]:
        pass


class SentenceSplitter(Pretokeniser):
    def splitSentence(self, text: str) -> List[str]:
        return [text]


class WordSplitter(Pretokeniser):
    """
    Splits on (and destroys all) whitespace, replacing it by a marker.
    """

    def __init__(self, output_marker: SpaceMarker):
        self.marker = output_marker

    def splitSentence(self, text: str) -> List[str]:
        results = []
        if not text.strip():
            return results

        if self.marker.location == MarkerLocation.TOKEN:
            results = text.split()  # Will strip all whitespace from both sides, then split on any span of whitespace.
            results = intercalate(results, self.marker.substitute)  # Will have length 2n-1.
            if text[0].isspace():
                results.insert(0, self.marker.substitute)
            if text[-1].isspace():  # Due to the sanity check above, we know that this is not the same whitespace!
                results.append(self.marker.substitute)

        elif self.marker.location == MarkerLocation.START:
            pass  # TODO: Implement SoW

        elif self.marker.location == MarkerLocation.END:
            pass  # TODO: Implement EoW

        return results

    def splitWord(self, word: str) -> List[str]:
        """
        Extra method for algorithms like BPE that require a word to start out as
        being split into characters.
        """
        if self.marker.location == MarkerLocation.START:
            pass  # TODO
        elif self.marker.location == MarkerLocation.END:
            pass  # TODO
        else:
            return list(word)


def intercalate(lst: list, new_element):
    new_list = []
    for old_element in lst:
        new_list.append(old_element)
        new_list.append(new_element)
    return new_list[:-1]