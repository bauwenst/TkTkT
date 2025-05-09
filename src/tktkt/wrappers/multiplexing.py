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
"""
from typing import List, Iterable
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.random as npr

from ..interfaces.tokeniser import Tokeniser, Preprocessor, TokeniserWithFiniteTypeDomain


@dataclass
class MultiplexedPreprocessor:
    global_preprocessor: Preprocessor  # This preprocessor's pretokens are sent to different tokenisers. Recommended: a simple whitespace tokeniser.
    specific_preprocessors: bool


class TokeniserMultiplexer(Tokeniser):

    def __init__(self, preprocessor: MultiplexedPreprocessor, subtokenisers: List[Tokeniser]):
        assert subtokenisers
        super().__init__(preprocessor=preprocessor.global_preprocessor)
        self.subtokenisers = subtokenisers
        self._use_specific_preprocessors = preprocessor.specific_preprocessors

    @abstractmethod
    def select(self, pretoken: str) -> int:
        pass

    def tokenise(self, pretoken: str) -> List[str]:
        subtokeniser = self.subtokenisers[self.select(pretoken)]
        # print(f"\tPretoken <{pretoken}> will be tokenised by {subtokeniser.getName()}")
        if self._use_specific_preprocessors:
            return subtokeniser.prepareAndTokenise(pretoken)
        else:
            return subtokeniser.tokenise(pretoken)

    def getName(self) -> str:
        return "Multiplex(" + " + ".join(sub.getName() for sub in self.subtokenisers) + ")"


class TokeniserMultiplexer_SameDomains(TokeniserMultiplexer, TokeniserWithFiniteTypeDomain):
    """
    Multiplexer where all the multiplexed tokenisers share the same domain-to-range mapping.
    Note: domain and range should be small enough to be enumerated into a set.

    Calls the super constructor of TokeniserMultiplexer while keeping its select() method abstract;
    implements all the abstract methods of TokeniserWithFiniteTypeDomain.
    """
    def __init__(self, preprocessor: MultiplexedPreprocessor, subtokenisers: List[TokeniserWithFiniteTypeDomain]):
        super().__init__(preprocessor, subtokenisers)

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


class TokeniserMultiplexer_DifferentDomains(TokeniserMultiplexer, TokeniserWithFiniteTypeDomain):
    """
    Similar to TokeniserMultiplexer_SameDomains, except now all the subtokenisers have their own domains and IDs are
    obtained by offsetting the IDs of subtokeniser i by the size of the domain of subtokenisers 1 to i-1.
    """

    def __init__(self, preprocessor: MultiplexedPreprocessor, subtokenisers: List[TokeniserWithFiniteTypeDomain]):
        super().__init__(preprocessor, subtokenisers)
        self.subtokenisers: List[TokeniserWithFiniteTypeDomain] = self.subtokenisers  # (type casting)

        # Verify that all IDs in the subtokenisers run from 0 to |V_i|-1.
        vocab_sizes = []
        for tokeniser in subtokenisers:
            V = tokeniser.getVocabSize()
            vocab_sizes.append(V)
            assert set(tokeniser.ids()) == set(range(V))

        # Offsets is all you need.
        self._offsets = np.cumsum([0] + vocab_sizes[:-1]).tolist()

    def tokenise(self, pretoken: str) -> List[str]:
        idx = self.select(pretoken)
        subtokeniser = self.subtokenisers[idx]
        if self._use_specific_preprocessors:
            tokens = subtokeniser.prepareAndTokenise(pretoken)
        else:
            tokens = subtokeniser.tokenise(pretoken)
        return [f"{idx}_{token}" for token in tokens]  # The only way to know which token was produced by which tokeniser is to modify the tokens, since conversion from string to integer is done long after many subtokenisers have been applied to all input in the batch.

    def types(self) -> Iterable[str]:
        for idx, subtokeniser in enumerate(self.subtokenisers):
            for typ in subtokeniser.types():
                yield f"{idx}_{typ}"

    def idToType(self, i: int) -> str:
        idx = 0
        while idx < len(self.subtokenisers) and i >= self._offsets[idx]:
            idx += 1
        idx -= 1

        return f"{idx}_{self.subtokenisers[idx].idToType(i - self._offsets[idx])}"

    def ids(self) -> Iterable[int]:
        for subtokeniser, offset in zip(self.subtokenisers, self._offsets):
            for i in subtokeniser.ids():
                yield offset + i

    def typeToId(self, t: str) -> int:
        idx, t = t.split("_", maxsplit=1)
        idx = int(idx)
        return self._offsets[idx] + self.subtokenisers[idx].typeToId(t)


class CompressiveTokeniserMultiplexer(TokeniserMultiplexer):
    def select(self, pretoken: str) -> int:
        return min(range(len(self.subtokenisers)), key=lambda i: len(self.subtokenisers[i].prepareAndTokenise(pretoken)))


class StochasticTokeniserMultiplexer(TokeniserMultiplexer):
    """
    Sample tokenisers proportional according to a given probability mass.
    """
    def __init__(self, preprocessor: MultiplexedPreprocessor,
                 subtokenisers: List[Tokeniser], probabilities: List[float]=None, seed: int=0):
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

    def select(self, pretoken: str) -> int:
        return self._rng.choice(self._n, p=self._distribution)  # .choice() because sadly, .integers() has no probability mass argument.


class StochasticTokeniserMultiplexer_SameDomains(StochasticTokeniserMultiplexer, TokeniserMultiplexer_SameDomains):
    """
    Takes its .select() implementation from StochasticTokeniserMultiplexer, and implementations for methods that have to
    do with the token domain from TokeniserMultiplexer_SameDomains.
    """
    def __init__(self, preprocessor: MultiplexedPreprocessor, subtokenisers: List[TokeniserWithFiniteTypeDomain],
                 probabilities: List[float]=None, seed: int=0):
        super().__init__(preprocessor, subtokenisers, probabilities, seed)
        TokeniserMultiplexer_SameDomains.__init__(self, preprocessor, subtokenisers)


class StochasticTokeniserMultiplexer_DifferentDomains(StochasticTokeniserMultiplexer, TokeniserMultiplexer_DifferentDomains):
    """
    Takes its .select() implementation from StochasticTokeniserMultiplexer, and implementations for methods that have to
    do with the token domain from TokeniserMultiplexer_DifferentDomains.
    """
    def __init__(self, preprocessor: MultiplexedPreprocessor, subtokenisers: List[TokeniserWithFiniteTypeDomain],
                 probabilities: List[float]=None, seed: int=0):
        super().__init__(preprocessor, subtokenisers, probabilities, seed)
        TokeniserMultiplexer_DifferentDomains.__init__(self, preprocessor, subtokenisers)


class StochasticTokeniserSwitch(StochasticTokeniserMultiplexer_SameDomains):
    """
    Multiplexer that only takes two tokenisers (with the same domain) and uses a more efficient sampler than
    npr.choice() to choose the tokeniser.
    """

    def __init__(self, preprocessor: MultiplexedPreprocessor,
                 tokeniser1: TokeniserWithFiniteTypeDomain, tokeniser2: TokeniserWithFiniteTypeDomain, p: float=0.5):
        """
        :param p: Probability of sampling tokeniser 2. This way, the [0,1] interval is a slider that ranges
                  from always tokeniser 1 to always tokeniser 2.
        """
        super().__init__(preprocessor, [tokeniser1, tokeniser2], probabilities=[(1-p), p])
        self.threshold = p

    def select(self, pretoken: str) -> int:
        return self._rng.random() < self.threshold
