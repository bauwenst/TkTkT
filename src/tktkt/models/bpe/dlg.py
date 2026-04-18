"""
Uses description length gain (DLG) as the argmax objective for pairs.
https://aclanthology.org/W99-0701
"""
from pathlib import Path
from math import log2

from pickybpe.util.counters import *
from pickybpe.vocabularisation import Token

from .vocabularisation import Preprocessor, CacheableBPEArtifacts, _VocabulariserWithChizhovBackend, \
    _ChizhovTrainingContext, _ChizhovBackend_BPE
from .statistical import CountingObjective, Pair


class _DLGBPEObjective(CountingObjective):

    def __init__(self):
        self._pair_counts: FlatCounter[Pair] = FlatCounter()
        self._pair_metrics: FlatCounterArgmaxable[Pair] = FlatCounterArgmaxable()

    @property
    def counts(self):
        return self._pair_counts

    def has(self, pair: Pair) -> bool:
        return self._pair_counts.has(pair) and self._pair_metrics.has(pair)

    def pop(self, pair: Pair) -> tuple[int, float]:
        count = self._pair_counts.pop(pair)
        metric = self._pair_metrics.pop(pair)
        return count, metric

    def get_argmax_objective(self) -> Pair:
        return self._pair_metrics.get_argmax()[0]

    def recompute_objective(self, pairs_with_updated_counts: Iterable[Pair], state: _ChizhovTrainingContext, subtokens: Optional[Pair]):
        """
        The formula for DLG, namely

            DLG(x,y) = DL - DL_{x,y}
                     = |T_n| H_n - |T_{n+1}| H_{n+1}
                     = sum_{x in V}        -C_n(x)     log C_n(x)     / |T_n|
                     - sum_{x in V u {xy}} -C_{n+1}(x) log C_{n+1}(x) / |T_{n+1}|

        can be computed in O(1) time and, even better, for big corpora does not require recomputation across all pairs
        every iteration, because the summation evaporates.

        Since C_n(t) == C_{n+1}(t) for all t that are not in {x,y,xy,SEP} and since we know C_n(SEP) = n, we get

            DLG(x,y) = C_{n+1}(x) log C_{n+1}(x) / |T_{n+1}| - C_n(x) log C_n(x) / |T_n|
                     + C_{n+1}(y) log C_{n+1}(y) / |T_{n+1}| - C_n(y) log C_n(y) / |T_n|
                     + C_{n+1}(xy) log C_{n+1}(xy) / |T_{n+1}|
                     + (n+1) log (n+1) / |T_{n+1}| - n log n / |T_n|
                     + sum_{t in V'} C_n(t) ( log C_n(t) / |T_{n+1}| - log C_n(t) / |T_n| )

        where the last term equals

                     (|T_n| - C_n(x) - C_n(y) - n) * log |T_n| / |T_{n+1}|
                    = (|T_n| - C_n(x) - C_n(y) - n) * log |T_n|
                    - (|T_n| - C_n(x) - C_n(y) - n) * log |T_{n+1}|
                    = |T_n| log |T_n|
                    - (|T_n| - C_n(x) - C_n(y) - n) * log |T_{n+1}|
                    - (C_n(x) + C_n(y) + n) * log |T_n|

        which is no longer O(|V|) to compute. The last term takes away the |T_n| denominator in all the logarithms above.
        What remains is

            DLG(x,y) = C_{n+1}(x) log C_{n+1}(x) / |T_{n+1}| - C_n(x) log C_n(x)
                     + C_{n+1}(y) log C_{n+1}(y) / |T_{n+1}| - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy) / |T_{n+1}|
                     +       (n+1) log (n+1) / |T_{n+1}|     - n log n
                     - (|T_n| - C_n(x) - C_n(y) - n) * log |T_{n+1}|
                     + |T_n| log |T_n|
                     = C_{n+1}(x) log C_{n+1}(x) - C_n(x) log C_n(x)
                     + C_{n+1}(y) log C_{n+1}(y) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     + (n+1) log (n+1)           - n log n
                     - (C_{n+1}(x) + C_{n+1}(y) + C_{n+1}(xy) + (n+1)) * log |T_{n+1}|
                     - (|T_n| - C_n(x) - C_n(y) - n)                   * log |T_{n+1}|
                     + |T_n| log |T_n|
                     = C_{n+1}(x) log C_{n+1}(x) - C_n(x) log C_n(x)
                     + C_{n+1}(y) log C_{n+1}(y) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     - (|T_n| + C_{n+1}(x) - C_n(x) + C_{n+1}(y) - C_n(y) + C_{n+1}(xy) + 1) * log |T_{n+1}|
                     + |T_n| log |T_n| + (n+1) log (n+1) - n log n
                     = C_{n+1}(x) log C_{n+1}(x) - C_n(x) log C_n(x)
                     + C_{n+1}(y) log C_{n+1}(y) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     - (|T_n| + Δ_x + Δ_y + C_n(x,y) + 1) * log |T_{n+1}|
                     + constant

        with the constant independent of x and y. Originally, I concluded from this:

            For really large |T_n|, the last term reduces to -|T_n| log |T_{n+1}|
            and this is again constant, so we end up with

                DLG(x,y) = C_{n+1}(x)  log C_{n+1}(x) - C_n(x) log C_n(x)
                         + C_{n+1}(y)  log C_{n+1}(y) - C_n(y) log C_n(y)
                         + C_{n+1}(xy) log C_{n+1}(xy)
                         + constant

            which only needs to be recomputed when x or y or (x,y) changes counts, not necessarily when |T_n| changes. Hence
            a heap is appropriate.

        You can test this and it will give garbage results. The reason it can't be right is that we are saying that a
        term which stops existing, i.e. (|T_n|+1) * log |T_{n+1}|, removes a term which wouldn't stop existing if that
        other term never existed, namely (C_n(x,y) + Δ_x + Δ_y) * log |T_{n+1}|. This is a grandfather paradox.
        In fact, the second term is much larger than the remaining terms, e.g.
            C_{n+1}(xy) log C_{n+1}(xy) << C_{n+1}(xy) * log( |T_n| + C_{n+1}(xy) ).
        So we need to go deeper.

        Credit goes to Claude AI for suggesting to cancel the logarithms against each other using a Taylor expansion:
        taking log2(x + Δ) ~ log2(x) + 1/(ln2) * 1/x * Δ we get

            DLG(x,y) = C_{n+1}(x) log C_{n+1}(x) - C_n(x) log C_n(x)
                     + C_{n+1}(y) log C_{n+1}(y) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     - (|T_n| + Δ_x + Δ_y + C_n(x,y) + 1) * log |T_{n+1}|
                     = C_{n+1}(x) * (log C_n(x) + 1/(ln2) * 1/C_n(x) * Δ_x) - C_n(x) log C_n(x)
                     + C_{n+1}(y) * (log C_n(y) + 1/(ln2) * 1/C_n(y) * Δ_y) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     - (|T_n| + Δ_x + Δ_y + C_n(x,y) + 1) * (log |T_n| + Δ * 1/(ln2) * 1/|T_n|)
                     = (C_n(x) + Δ_x) * (log C_n(x) + Δ_x/(ln2 * C_n(x)) ) - C_n(x) log C_n(x)
                     + (C_n(y) + Δ_y) * (log C_n(y) + Δ_y/(ln2 * C_n(y)) ) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     - (|T_n| + Δ_x + Δ_y + C_n(x,y) + 1) * (log |T_n| + Δ/(ln2 * |T_n|) )
                     = Δ_x * (log C_n(x) + Δ_x/(ln2 * C_n(x)) ) + Δ_x/ln2
                     + Δ_y * (log C_n(y) + Δ_y/(ln2 * C_n(y)) ) + Δ_y/ln2
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     - |T_n| log |T_n| - Δ/ln2
                     - (Δ_x + Δ_y + C_n(x,y) + 1) * (log |T_n| + Δ/(ln2 * |T_n|) )
                     = Δ_x * (log C_n(x) + Δ_x/(ln2 * C_n(x)) + 1/ln2)
                     + Δ_y * (log C_n(y) + Δ_y/(ln2 * C_n(y)) + 1/ln2)
                     + C_n(x,y) log C_n(x,y)
                     - Δ/ln2
                     - (Δ_x + Δ_y + C_n(x,y) + 1) * (log |T_n| + Δ/(ln2 * |T_n|) )
                     + constant
                     = Δ_x * (log C_n(x) + Δ_x/(ln2 * C_n(x)) + 1/ln2)
                     + Δ_y * (log C_n(y) + Δ_y/(ln2 * C_n(y)) + 1/ln2)
                     + C_n(x,y) log C_n(x,y)
                     - Δ/ln2 - Δ/(ln2 * |T_n|)
                     -      Δ_x * (log |T_n| + Δ/(ln2 * |T_n|) )
                     -      Δ_y * (log |T_n| + Δ/(ln2 * |T_n|) )
                     - C_n(x,y) * (log |T_n| + Δ/(ln2 * |T_n|) )
                     + constant
                     =      Δ_x * (log C_n(x)   - (ln2*log |T_n| + Δ/|T_n| )/ln2 + Δ_x/(ln2 * C_n(x)) + 1/ln2 )
                     +      Δ_y * (log C_n(y)   - (ln2*log |T_n| + Δ/|T_n| )/ln2 + Δ_y/(ln2 * C_n(y)) + 1/ln2 )
                     + C_n(x,y) * (log C_n(x,y) - (ln2*log |T_n| + Δ/|T_n| )/ln2)
                     - Δ/ln2 - Δ/(ln2 * |T_n|)
                     + constant
                     ~      Δ_x * (ln C_n(x)   - (ln |T_n| + Δ/|T_n|) + Δ_x/C_n(x) + 1)
                     +      Δ_y * (ln C_n(y)   - (ln |T_n| + Δ/|T_n|) + Δ_y/C_n(y) + 1)
                     + C_n(x,y) * (ln C_n(x,y) - (ln |T_n| + Δ/|T_n|))
                     - Δ * (1 + 1/|T_n|)

        Now we will finally drop terms by assuming |T_n| is really big, and then
        substitute Δ_x = -m_x * (C_n(x,y)-1) and Δ = C_n(x,y) + Δ_x + Δ_y + 1.

                DLG(x,y) ~~     Δ_x * (ln C_n(x)   - ln |T_n| + Δ_x/C_n(x) + 1)
                         +      Δ_y * (ln C_n(y)   - ln |T_n| + Δ_y/C_n(y) + 1)
                         + C_n(x,y) * (ln C_n(x,y) - ln |T_n|)
                         - Δ
                         =      Δ_x * (ln C_n(x)   - ln |T_n| + Δ_x/C_n(x)) + Δ_x
                         +      Δ_y * (ln C_n(y)   - ln |T_n| + Δ_y/C_n(y)) + Δ_y
                         + C_n(x,y) * (ln C_n(x,y) - ln |T_n|)
                         - (C_n(x,y) + Δ_x + Δ_y + 1)
                         = Δ_x * (ln C_n(x)/|T_n| + Δ_x/C_n(x))
                         + Δ_y * (ln C_n(y)/|T_n| + Δ_y/C_n(y))
                         + C_n(x,y) * ln C_n(x,y)/|T_n|
                         - C_n(x,y) + 1
                         = C_n(x,y) ln C_n(x,y)/|T_n|
                         - C_n(x,y)
                         - m_x * (C_n(x,y)-1) * (ln C_n(x)/|T_n| + Δ_x/C_n(x))
                         - m_y * (C_n(x,y)-1) * (ln C_n(y)/|T_n| + Δ_y/C_n(y))
                         + constant
                         = C_n(x,y) ln C_n(x,y)/|T_n|
                         - C_n(x,y)
                         - C_n(x,y) ln (C_n(x)/|T_n|)^m_x
                         - C_n(x,y) ln (C_n(y)/|T_n|)^m_y
                         - m_x * ( Δ_x*C_n(x,y)/C_n(x) - 1*(ln C_n(x)/|T_n| + Δ_x/C_n(x)))
                         - m_y * ( Δ_y*C_n(x,y)/C_n(y) - 1*(ln C_n(y)/|T_n| + Δ_y/C_n(y)))
                         + constant
                         = C_n(x,y) * (PMI(x,y) - 1)
                         - m_x * ( Δ_x*C_n(x,y)/C_n(x) - ln C_n(x)/|T_n| - Δ_x/C_n(x))
                         - m_y * ( Δ_y*C_n(x,y)/C_n(y) - ln C_n(y)/|T_n| - Δ_y/C_n(y))
                         + constant
                         = C_n(x,y) * (PMI(x,y) - 1)
                         - m_x * ( (C_n(x,y)-1) * Δ_x/C_n(x) - ln C_n(x)/|T_n| )
                         - m_y * ( (C_n(x,y)-1) * Δ_y/C_n(y) - ln C_n(y)/|T_n| )
                         + constant
                         = C_n(x,y) * (PMI(x,y) - 1)
                         + Δ_x²/C_n(x) + m_x ln C_n(x)/|T_n|
                         + Δ_y²/C_n(y) + m_y ln C_n(y)/|T_n|
                         + constant

        although one could argue that the logarithm isn't really a true PMI here since C_n(x,y) is counted respecting the
        pretoken boundaries. Normally if you sum C_n(x,y)/(|T_n|-1) across all bigrams, you get 1 because there are |T_n|-1
        bigrams in a sequence of |T_n| tokens. Clearly this is not the case here, so the operator called "PMI" above
        has a slightly different range than PMI normally does since the numerators across all pairs don't sum to anywhere close to |T_n|-1.

        The terms with m_i actually form the denominator of PMI, but can't combine with it since the PMI is multiplied by C_n(x,y).
        I guess you could simplify them to "PMI without the numerator" though:

             DLG(x,y) ~~ C_n(x,y) * (PMI(x,y) - 1)
                      - ln 1/[(C_n(x)/|T_n|)^m_x * (C_n(y)/|T_n|)^m_y]
                      + Δ_x²/C_n(x) + Δ_y²/C_n(y)
                      = C_n(x,y) * (PMI(x,y) - 1)
                      - (PMI(x,y) - ln C_n(x,y)/|T_n|)
                      + Δ_x²/C_n(x) + Δ_y²/C_n(y)
                      = (C_n(x,y)-1) * PMI(x,y) - C_n(x,y)
                      + ln C_n(x,y)/|T_n|
                      + Δ_x²/C_n(x) + Δ_y²/C_n(y)

        I'm not sure if you can make the simplification C_n(x,y)-1 ~ C_n(x,y), but it would give

            DLG(x,y) ~~ = C_n(x,y) * (PMI(x,y) - 1)
                        + ln C_n(x,y)/|T_n|
                        + Δ_x²/C_n(x) + Δ_y²/C_n(y)

        Interestingly, PMI is very much dependent on |T_n|, so C(x,y)*PMI(x,y) is also impossible to heapify usefully.
        (Also, to compute true DLG, you should add the constant we left out above, which we only left out because we
        thought we could avoid |T_n| and since we can't, there is no use leaving the constant out anymore except for
        the elegance of the formula.)

        Really, if you want the precise DLG formula, you should just stick to the result we got long ago and not
        simplify further, since it can be computed exactly:

            DLG(x,y) = (C_n(x) + Δ_x) log(C_n(x) + Δ_x) - C_n(x) log C_n(x)
                     + (C_n(y) + Δ_y) log(C_n(y) + Δ_y) - C_n(y) log C_n(y)
                     + (C_n(x,y)    ) log(C_n(x,y)    )
                     -((|T_n|  + Δ  ) log(|T_n| + Δ   ) -  |T_n| log |T_n| )
                     +          (n+1) log (n+1)         -      n log n

        You can even normalise this by C_n(x,y) if you want, which apparently works as scores for a Viterbi tokeniser.
        """
        # Old implementation: only pairs related to the latest merge are updated.
        # pairs = set(pairs_with_updated_counts)
        # for token in subtokens or []:  # Doesn't run at initialisation.
        #     pairs |= {pair for pair in pairs_with_token(token) if self.has(pair)}  # The reason you need this has() call is that pairs_with_token sources its pairs from Word objects, which only contain pairs that went through validation. The pairs that passed that validation are in the heap, the ones that failed the validation are not. So you can use self.counts.has() as equivalent for trainer._validate_pair().
        # for pair in pairs:
        # New implementation: every pair is updated, regardless of the latest merge.
        C = state.corpus_token_count
        n = len(state.events)
        for pair in self._pair_counts:
            L = len(pair)  # Sum over all m's
            C_xy = self._pair_counts.get(pair)
            D    = (C_xy + 1) + (L - L*C_xy)  # Sum over all D_x's, plus C_xy plus 1. How many new tokens do we get? We merge into many xy tokens, we add a comma, and we add a merge rule with its parts. But, we lose all of the parts for each merge.
            score =     C_xy * log2(C_xy) \
                  +  (n + 1) * log2(n + 1) - n * log2(n or 1) \
                  - ((C + D) * log2(C + D) - C * log2(C) )
            for token in set(pair):  # The set() is because the DLG formula is a sum over the vocabulary, i.e. unique tokens.
                m    = pair.count(token)
                C_x  = token.freq
                D_x  = m - m*C_xy  # The +m is because in DLG, you assume that the merge itself is appended to the end of the corpus and thus all its C_xy occurrences disappear and 1 new one appears.
                score += (C_x + D_x) * log2(C_x + D_x) - C_x * log2(C_x)
            self._pair_metrics.set(pair, score)


class _ChizhovBackend_DLGBPE(_ChizhovBackend_BPE):  # Inherits the preprocessing and dumping logic.

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
        super().__init__(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            max_type_length=max_type_length,
        )

    def _initialize_state(self) -> _ChizhovTrainingContext:
        state = super()._initialize_state()
        state.pairs = _DLGBPEObjective()
        return state


class DLGBPEVocabulariser(_VocabulariserWithChizhovBackend[CacheableBPEArtifacts]):  # Analogous to BPEVocabulariser_Chizhov
    """
    BPE variant using description length gain (DLG) as the argmax objective for pairs.
    https://aclanthology.org/W99-0701
    """

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int):
        super().__init__(preprocessor=preprocessor, backend=_ChizhovBackend_DLGBPE(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            max_type_length=max_type_length,
            character_coverage=character_coverage
        ))

    def _cacheSubfolder(self) -> str:
        return "dlg-bpe"

    def _cacheType(self):
        return CacheableBPEArtifacts

    def _dumpToArtifacts(self, dump_path: Path) -> CacheableBPEArtifacts:
        return CacheableBPEArtifacts(
            types=CacheableBPEArtifacts._loadTypes(dump_path),
            merges=CacheableBPEArtifacts._loadMerges(dump_path)
        )


def pairs_with_token(token: Token) -> set[Pair]:
    return {pair for word in token.words for pair in word.pairs if token in pair}
