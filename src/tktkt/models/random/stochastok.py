import numpy.random as npr

from ...interfaces.tokeniser import *
from ...wrappers.multiplexing import SuccessionalTokeniser


class StochastTok(TokeniserWithVocabulary[WithSpecials], SuccessionalTokeniser):
    """
    Uses a base tokeniser to segment the input, and then randomly breaks down the result further into tokens
    part of the vocabulary.

    There are multiple ways to do this. In particular, which tokens to apply this to (i.e. when to stop), and in which
    order to apply this.
        - Stopping conditions:
            1. Geometrically: after each application, decide whether you stop or not.
            2. Deterministically: predetermine how many applications to do, e.g. proportional to the length of the original sequence.
            3. Binomially: decide per token if you perturb it or not. (Expected value equals the deterministic case.)
        - Order:
            1. Within-token: keep splitting one token and its descendants before splitting the next one.
            2. Token-by-token: sample a token, split it, repeat.
            3. Sequence-by-sequence: sample tokens from the sequence, split them all, repeat.

    StochasTok by Sims et al. (2025) (https://arxiv.org/abs/2506.01687) uses deterministic stopping and token-by-token
    ordering, with the extra caveat that when a token is sampled and it cannot be split, it still counts.
    """
    def __init__(self, base: TokeniserWithVocabulary[WithSpecials], proportion: float, seed: int=0, accelerate: bool=True):
        super().__init__(preprocessor=base.preprocessor, vocab=base.vocab)
        self._base = base
        self._rng  = npr.default_rng(seed=seed)
        self._p = proportion

        self._accelerated = accelerate
        if accelerate:
            self._possible_splits = {
                t: [i for i in range(len(t)-1) if self.hasType(t[:i+1]) and self.hasType(t[i+1:])]
                for t in self.types()
            }
        else:
            self._possible_splits = dict()

    def _initialTokens(self, pretoken: str) -> Tokens:
        return self._base.tokenise(pretoken)

    def _finalTokens(self, tokens: Tokens) -> Tokens:
        tokens = list(tokens)

        budget = int(len(tokens)*self._p)
        for _ in range(budget):
            token_idx = self._rng.integers(len(tokens))
            token = tokens[token_idx]
            possible_splits = self._possible_splits[token] if self._accelerated else [i for i in range(len(token)-1) if self.hasType(token[:i+1]) and self.hasType(token[i+1:])]
            if not possible_splits:
                continue
            split_idx = self._rng.choice(possible_splits)
            tokens[token_idx] =        token[:split_idx+1]
            tokens.insert(token_idx+1, token[split_idx+1:])

        return tokens
