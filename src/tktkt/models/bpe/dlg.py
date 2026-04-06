"""
Uses description length gain (DLG) as the argmax objective for pairs.
https://aclanthology.org/W99-0701
"""
from pathlib import Path
from math import log2

from pickybpe.util.counters import *

from .vocabularisation import Preprocessor, CacheableBPEArtifacts, _VocabulariserWithChizhovBackend, \
    _ChizhovTrainingContext, _ChizhovBackend_BPE
from .statistical import pairs_with_token, CountingObjective, Pair


class _DLGBPEObjective(CountingObjective):

    def __init__(self):
        self._pair_counts: FlatCounter[Pair] = FlatCounter()
        self._pair_metrics: MaxHeap[Pair] = MaxHeap()

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
                     - (|T_n| + Delta_x + Delta_y + C_n(x,y) + 1) * log |T_{n+1}|
                     + constant

        with the constant independent of x and y. For really large |T_n|, the last term reduces to -|T_n| log |T_{n+1}|
        and this is again constant, so we end up with

            DLG(x,y) = C_{n+1}(x)  log C_{n+1}(x) - C_n(x) log C_n(x)
                     + C_{n+1}(y)  log C_{n+1}(y) - C_n(y) log C_n(y)
                     + C_{n+1}(xy) log C_{n+1}(xy)
                     + constant

        which only needs to be recomputed when x or y or (x,y) changes counts, not necessarily when |T_n| changes. Hence
        a heap is appropriate.
        """
        pairs = set(pairs_with_updated_counts)
        for token in subtokens:
            pairs |= pairs_with_token(token)
        for pair in pairs:
            C_xy = self._pair_counts.get(pair)
            score = C_xy * log2(C_xy)
            for token in set(pair):
                m    = pair.count(token)
                C_x  = token.freq
                D_x  = -m*(C_xy-1)  # The -1 is because in DLG, you assume that the merge itself is appended to the end of the corpus and thus all its C_xy occurrences disappear and 1 new one appears.
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
