from math import log as ln

from pickybpe.vocabularisation import CountingObjective, Pair, MovingAverage
from pickybpe.util.counters import *

from .vocabularisation import Preprocessor, _ChizhovBackend_BPE, _ChizhovTrainingContext


class SBPEObjective(CountingObjective):

    def __init__(self):
        self._pair_counts: FlatCounter[Pair] = FlatCounter()
        self._pair_metrics: MaxHeap[Pair] = MaxHeap()

    @property
    def counts(self):
        return self._pair_counts

    def has(self, pair: Pair) -> bool:
        return self._pair_counts.has(pair) and self._pair_metrics.has(pair)

    def pop(self, pair: Pair) -> tuple[int,float]:
        count  = self._pair_counts.pop(pair)
        metric = self._pair_metrics.pop(pair)
        return count, metric

    def get_argmax_objective(self) -> Pair:
        return self._pair_metrics.get_argmax()[0]

    def recompute_objective(self, pairs: Iterable[Pair], state: _ChizhovTrainingContext):
        """
        Equation 6 in the paper shows the score formula

            C_n(xy) * ln( P_n(xy) / ( P_n(x)P_n(y) ) )

        where C is a count, P is a probability, x and y and xy are types, and 'n' is the hypothetical next time step after
        merging pairs (x,y) into xy. What we have are the counts for all tokens and all pairs at iteration n-1.

        The probabilities are computed with MLE estimators to be

            C_n(xy) * ln( C_n(xy)/|T_n|  / ( C_n(x)/|T_n| * C_n(y)/|T_n| ) )
            = C_n(xy) * ln( |T_n| * C_n(xy)  / ( C_n(x) * C_n(y) ) )
            = C_n(xy) * [ ln C_n(xy) - ln C_n(x) - ln C_n(y) + ln |T_n| ]

        wherein T is the list resulting from tokenise(corpus) and furthermore
            C_n(xy) = C_{n-1}(x,y) when x != y;
            |T_n| =   |T_{n-1}| - C_n(xy) = |T_{n-1}|  - C_{n-1}(x,y);
            C_n(x) = C_{n-1}(x) - C_n(xy) = C_{n-1}(x) - C_{n-1}(x,y);
            C_n(y) = C_{n-1}(y) - C_n(xy) = C_{n-1}(y) - C_{n-1}(x,y);

        except the probabilities are computed using Laplace smoothing of the MLEs, so C'_n(t) = C_n(t) + 1 and |T'_n| = |T_n| + |V_n| = |T_n| + (|V_{n-1}| + 1).
        """
        for pair in pairs:
            # Count tokens in the hypothetical corpus where the pair is merged.
            count_pair   = self._pair_counts.get(pair)
            count_first  = pair[0].freq - count_pair
            count_second = pair[1].freq - count_pair
            sum_count    = state.corpus_token_count - count_pair

            # Do Laplace smoothing (add 1 to each type's count and add |V_n| = |V_{n-1}|+1 to the sum of type counts)
            # and unpack the probability ratios into a sum of logarithms.
            score = count_pair * (
                ln(count_pair   + 1)
              - ln(count_first  + 1)
              - ln(count_second + 1)
              + ln(sum_count + (state.actual_vocab_size + 1))
            )

            self._pair_metrics.set(pair, score)


class _ChizhovBackend_SBPETrainer(_ChizhovBackend_BPE):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int, character_coverage: float, max_type_length: int,
                 moving_average_threshold: float=0.002, moving_average_width: int=100, moving_average_stride: int=100):
        super().__init__(
            preprocessor=preprocessor,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            max_type_length=max_type_length,
        )
        self._threshold = moving_average_threshold
        self._delta     = MovingAverage(width=moving_average_width, stride=moving_average_stride)
        self._delta_0: float = 0

    def _initialize_state(self) -> _ChizhovTrainingContext:
        state = super()._initialize_state()
        state.pairs = SBPEObjective()
        return state

    def _stopping_condition(self, latest_score: float, state: _ChizhovTrainingContext) -> bool:
        self._delta.add(latest_score)
        if not self._delta.ready():  # Nothing can be done this iteration.
            return False
        else:  # You can compute a new delta.
            delta = self._delta.compute()
            if not self._delta_0:
                self._delta_0 = delta
                return False
            else:
                return delta < self._threshold*self._delta_0
