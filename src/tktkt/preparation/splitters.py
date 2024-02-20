from typing import List, Tuple
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
            results = text.split()
            for i in range(len(results)):
                if i != 0 or text[0].isspace():
                    results[i] = self.marker.substitute + results[i]

        elif self.marker.location == MarkerLocation.END:
            results = text.split()
            for i in range(len(results)):
                if i != len(results)-1 or text[-1].isspace():
                    results[i] = results[i] + self.marker.substitute

        return results

    def splitWord(self, pretoken: str) -> List[str]:
        """
        Extra method for algorithms like BPE that require a word to start out as
        being split into characters.

        Does NOT add SoW/EoW because this is already done when you split the sentence into words.
        What it might do, however, is attach the SoW/EoW to the adjacent character.
        """
        if self.marker.location == MarkerLocation.START:
            chars, sow = self.stripMarker(pretoken)
            if self.marker.detached:
                return [sow] + list(chars)
            else:
                return [sow + chars[0]] + list(chars[1:])
        elif self.marker.location == MarkerLocation.END:
            chars, eow = self.stripMarker(pretoken)
            if self.marker.detached:
                return list(chars) + [eow]
            else:
                return list(chars[:-1]) + [chars[-1] + eow]
        else:
            return list(pretoken)

    def stripMarker(self, pretoken: str) -> Tuple[str,str]:
        """
        Retrieve the part of a pretoken that isn't a space marker.
        """
        L = len(self.marker.substitute)
        if self.marker.location == MarkerLocation.START:
            root, marker = pretoken[L:], pretoken[:L]
        elif self.marker.location == MarkerLocation.END:
            root, marker = pretoken[:len(pretoken)-L], pretoken[len(pretoken)-L:]
        else:
            root, marker = pretoken, ""

        return root, marker


def intercalate(lst: list, new_element):
    new_list = []
    for old_element in lst:
        new_list.append(old_element)
        new_list.append(new_element)
    return new_list[:-1]