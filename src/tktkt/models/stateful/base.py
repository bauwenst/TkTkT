from abc import abstractmethod

from ...interfaces.tokenisers import Tokeniser


class StatefulTokeniser(Tokeniser):
    @abstractmethod
    def stepDownstream(self):
        """Advance the state of the tokeniser when a downstream application finishes one iteration of some kind of loop."""
        pass
