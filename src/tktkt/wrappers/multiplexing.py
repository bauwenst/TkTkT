"""
Wrapper that multiplexes more than one Tokeniser.

There are 5 possible configurations to handle preprocessing:

| Apply global preprocessor? | Give global pretokens of one example to same tokeniser? | Preprocess with selected subtokeniser? |
| ---                        | ---                                                     | ---                                    |
| yes                        | no                                                      | no                                     |
| yes                        | no                                                      | yes                                    |
| yes                        | yes                                                     | no                                     |
| yes                        | yes                                                     | yes                                    |
| no (== yes, Identity)      | yes+no (there only is one pretoken)                     | yes                                    |

Every yes-yes-X style is arguably mediocre (multiplexing at the sentence level).
The bottom one can be implemented as yes-no-yes.
That means we really only need support for yes-no-yes and yes-no-no. Hence, there is always a global preprocessor
(sometimes an identity) which determines the pretokens over which we multiplex tokenisers. Whether the subtokeniser
preprocesses extra is then just a boolean decision.

TODO: Rework the hierarchy to reflect this. It means all multiplexers can have the same parent.
"""
from typing import List, Iterable
from abc import abstractmethod

import numpy as np
import numpy.random as npr

from ..interfaces.tokeniser import Tokeniser, Preprocessor, TokeniserWithFiniteTypeDomain


class _TokeniserMultiplexer(Tokeniser):

    def __init__(self, preprocessor: Preprocessor, subtokenisers: List[Tokeniser]):
        assert subtokenisers
        super().__init__(preprocessor=preprocessor)
        self.subtokenisers = subtokenisers

    @abstractmethod
    def select(self) -> int:
        pass


class TokeniserMultiplexer_PreprocessThenMultiplex(_TokeniserMultiplexer):
    """
    Applies a single preprocessor, and then for each resulting pretoken, selects a tokeniser to segment it.
    """
    def __init__(self, preprocessor: Preprocessor, subtokenisers: List[Tokeniser]):
        super().__init__(preprocessor=preprocessor, subtokenisers=subtokenisers)

    def tokenise(self, pretoken: str) -> List[str]:
        return self.subtokenisers[self.select()].tokenise(pretoken)


class TokeniserMultiplexer_MultiplexThenPreprocess(_TokeniserMultiplexer):
    """
    For each sentence, selects a preprocessor-tokeniser pair and runs both.
    """
    def __init__(self, subtokenisers: List[Tokeniser]):
        super().__init__(preprocessor=None, subtokenisers=subtokenisers)  # Note: This means all accesses to .preprocessor are invalid.

    def prepareAndTokenise(self, text: str) -> List[str]:
        return self.subtokenisers[self.select()].prepareAndTokenise(text)

    def tokenise(self, pretoken: str) -> List[str]:
        raise RuntimeError("This multiplexer has no .tokenise() method, because it must ensure the right preprocessor has been used to generate the given pretoken.")


class StochasticTokeniserMultiplexer(TokeniserMultiplexer_PreprocessThenMultiplex):
    """
    Sample tokenisers proportional according to a given probability mass.
    """
    def __init__(self, preprocessor: Preprocessor, subtokenisers: List[Tokeniser], probabilities: List[float]=None, seed: int=0):
        super().__init__(preprocessor=preprocessor, subtokenisers=subtokenisers)

        if probabilities is None:
            probabilities = [1/len(subtokenisers) for _ in range(len(subtokenisers))]
        else:
            assert all(p >= 0 for p in probabilities)

            if len(probabilities) == len(subtokenisers) - 1:
                probabilities.append(1 - sum(probabilities))
            else:
                assert len(probabilities) == len(subtokenisers)
                total = sum(probabilities)
                probabilities = [p/total for p in probabilities]

        self._rng = npr.default_rng(seed)
        self._distribution = np.array(probabilities)
        self._n = len(self.subtokenisers)

    def select(self) -> int:
        return self._rng.choice(self._n, p=self._distribution)  # .choice() because sadly, .integers() has no probability mass argument.


class StochasticTokeniserMultiplexer_SharedDomain(StochasticTokeniserMultiplexer, TokeniserWithFiniteTypeDomain):
    """
    StochasticTokeniserMultiplexer where all the multiplexed tokenisers share the same domain-to-range mapping.
    Note: domain and range should be small enough to be enumerated into a set.
    """
    def __init__(self, preprocessor: Preprocessor, subtokenisers: List[TokeniserWithFiniteTypeDomain], probabilities: List[float]=None, seed: int=0):
        super().__init__(preprocessor, subtokenisers, probabilities, seed)

        # Check whether the first tokeniser's domain<->range mapping can be used as a stand-in for all others.
        ref_tokeniser = subtokenisers[0]
        domain = set(ref_tokeniser.types())
        rrange = set(ref_tokeniser.ids())
        for tokeniser in subtokenisers:
            # First check: test that the given tokeniser's domain and range are not bigger nor smaller than the reference, and that they are equal.
            assert domain == set(tokeniser.types())
            assert rrange == set(tokeniser.ids())

            # Second check: now that we know the domain and range match up, check that their mapping matches up.
            for t in domain:
                assert ref_tokeniser.typeToId(t) == tokeniser.typeToId(t)
            for i in rrange:
                assert ref_tokeniser.idToType(i) == tokeniser.idToType(i)

        # Use this for all method calls.
        self._domain_and_range = ref_tokeniser

    def typeToId(self, t: str) -> int:
        return self._domain_and_range.typeToId(t)

    def ids(self) -> Iterable[int]:
        return self._domain_and_range.ids()

    def idToType(self, i: int) -> str:
        return self._domain_and_range.idToType(i)

    def types(self) -> Iterable[str]:
        return self._domain_and_range.types()


class StochasticTokeniserSwitch(StochasticTokeniserMultiplexer_SharedDomain):
    """
    Multiplexer that only takes two tokenisers (with the same domain) and uses a more efficient sampler than
    npr.choice() to choose the tokeniser.
    """

    def __init__(self, preprocessor: Preprocessor, tokeniser1: TokeniserWithFiniteTypeDomain, tokeniser2: TokeniserWithFiniteTypeDomain, p: float=0.5):
        """
        :param p: Probability of sampling tokeniser 2. This way, the [0,1] interval is a slider that ranges
                  from always tokeniser 1 to always tokeniser 2.
        """
        super().__init__(preprocessor, [tokeniser1, tokeniser2], probabilities=[(1-p), p])
        self.threshold = p

    def select(self) -> int:
        return self._rng.random() < self.threshold
