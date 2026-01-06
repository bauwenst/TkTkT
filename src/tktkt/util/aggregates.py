from dataclasses import dataclass

from .printing import wprint


@dataclass
class Return_MicroMacro:
    micro: float
    macro: float


@dataclass
class Return_PrReF1:
    recall: float
    precision: float
    f1: float


class ConfusionMatrix:

    def __init__(self):
        self.total_tp        = 0
        self.total_predicted = 0
        self.total_relevant  = 0
        self.total           = 0

    def add(self, tp: int, predicted: int, relevant: int, total: int, weight: float=1):
        self.total_tp        += weight*tp
        self.total_predicted += weight*predicted
        self.total_relevant  += weight*relevant
        self.total           += weight*total

    def computePrReF1(self):
        precision = self.total_tp/self.total_predicted if self.total_predicted else 1.0
        recall    = self.total_tp/self.total_relevant  if self.total_relevant  else 1.0
        f1        = ConfusionMatrix.f1(precision, recall)
        return precision, recall, f1

    def compute(self):
        tp = self.total_tp
        fp = self.total_predicted - self.total_tp
        fn = self.total_relevant - self.total_tp
        tn = self.total - tp - fp - fn
        return tp, fp, tn, fn

    def display(self):
        tp, fp, tn, fn = self.compute()
        string = "        \tpredicted\n"    +\
                 "        \t  +  \t  -\n"   +\
                f"actual +\t {tp}\t {fn}\n" +\
                f"       -\t {fp}\t {tn}"
        wprint(string)

    def displayRePrF1(self, indent=0):
        P, R, F1 = self.computePrReF1()
        wprint("\t"*indent + "Precision:", P)
        print("\t"*indent + "Recall:   ", R)
        print("\t"*indent + "F1:       ", F1)

    @staticmethod
    def f1(precision: float, recall: float):
        return 2*(precision*recall)/(precision+recall)

    @staticmethod
    def computeMatrixMacroAverage(matrices: list["ConfusionMatrix"]) -> tuple[float, float, float]:
        """
        Computes the macro-average Pr, Re, F1 for a list of confusion matrices.

        Note: although the Pr, Re, F1 returned by .compute() are a micro-average, this method is not the macro-average
        equivalent of that. This is because .compute() is the micro-average over all added word segmentations, NOT over
        a list of matrices. It is impossible to reconstruct the macro-average over word segmentations because we don't store
        their separate Pr, Re, F1.
        """
        n = len(matrices)
        if n == 0:
            return (1.0, 1.0, 1.0)

        tuples = [matrix.computePrReF1() for matrix in matrices]
        precisions, recalls, f1s = zip(*tuples)
        return sum(precisions)/n, sum(recalls)/n, sum(f1s)/n

    @classmethod
    def fromPositivesNegatives(cls, tp: int, fp: int, tn: int, fn: int) -> "ConfusionMatrix":
        cm = ConfusionMatrix()
        cm.total_tp        = tp
        cm.total_relevant  = tp + fn
        cm.total_predicted = tp + fp
        cm.total = tp + fp + tn + fn
        return cm


class MicroMacro:

    def __init__(self):
        self._micro_num = 0
        self._micro_den = 0
        self._macro_num = 0
        self._macro_den = 0

    def add(self, num: float, den: float):
        self._micro_num += num
        self._micro_den += den
        self._macro_num += num / den
        self._macro_den += 1

    def compute(self) -> tuple[float, float]:
        return self._micro_num/self._micro_den if self._micro_den else float("inf") if self._micro_num else 1.0, \
               self._macro_num/self._macro_den if self._macro_den else float("inf") if self._macro_num else 1.0


class NestedMicroMacro:
    """
    For metrics that are ratios, you can take the ratio at three levels:
        - Over the entire dataset (a.k.a. micro)
        - Over each example in the dataset, and these ratios are then averaged across the dataset (macro-micro).
        - Over each instance in each example in the dataset, and these ratios are then either:
            - averaged across examples and those averages are averaged across the dataset (macro-macro);
            - averaged across the dataset after concatenating examples (micro-macro);

    Usually, we assume each example has one instance, but this need not be so, e.g. if you have a metric that computes
    a ratio per token, whereas the dataset consists of pretokens.
    """

    @dataclass
    class NestedMicroMacro:
        micro_micro: float  # sum_w( sum_t(N) ) / sum_w( sum_t(D) )
        macro_micro: float  # avg_w( sum_t(N) / sum_t(D) )
        micro_macro: float  # avg_t( N/D )
        macro_macro: float  # avg_w( avg_t(N/D) )

    def __init__(self):
        self._current_micro_num = 0
        self._current_micro_den = 0
        self._current_macro     = 0
        self._current_n_instances = 0

        self._total_micro_num = 0
        self._total_micro_den = 0
        self._total_macro     = 0

        self._total_local_micro = 0
        self._total_local_macro = 0

        self._total_n_instances = 0
        self._total_n_examples  = 0

    def add(self, num: float, den: float):
        self._current_micro_num += num
        self._current_micro_den += den
        self._current_macro     += num/den
        self._current_n_instances += 1

    def _currentMicro(self) -> float:
        return self._current_micro_num / self._current_micro_den

    def _currentMacro(self) -> float:
        return self._current_macro / self._current_n_instances

    def fence(self):
        """
        End the current example and start a new one.
        """
        self._total_micro_num   += self._current_micro_num
        self._total_micro_den   += self._current_micro_den
        self._total_macro       += self._current_macro
        self._total_n_instances += self._current_n_instances
        self._total_n_examples  += 1

        self._total_local_micro += self._currentMicro()
        self._total_local_macro += self._currentMacro()

        self._current_micro_num = 0
        self._current_micro_den = 0
        self._current_macro     = 0
        self._current_n_instances = 0

    def compute(self):
        return NestedMicroMacro.NestedMicroMacro(
            micro_micro=self._total_micro_num / self._total_micro_den,
            macro_micro=self._total_local_micro / self._total_n_examples,
            micro_macro=self._total_macro / self._total_n_instances,
            macro_macro=self._total_local_macro / self._total_n_examples
        )


# class MicroMacro:
#
#     @dataclass
#     class MicroMacro:
#         macro: float
#         micro: float
#
#     def __init__(self):
#         self._core = NestedMicroMacro()
#
#     def add(self, num: float, den: float):
#         self._core.add(num, den)
#
#     def compute(self):
#         return MicroMacro.MicroMacro(
#             micro=self._core._currentMicro(),
#             macro=self._core._currentMacro()
#         )


class NestedAverage:

    @dataclass
    class NestedAverage:
        micro: float
        macro: float

    def __init__(self):
        self._current_sum         = 0
        self._current_n_instances = 0

        self._total_sum           = 0
        self._total_local_average = 0
        self._total_n_instances = 0
        self._total_n_examples  = 0

    def add(self, x: float):
        self._current_sum         += x
        self._current_n_instances += 1

    def fence(self):
        """
        End the current example and start a new one.
        """
        self._total_sum           += self._current_sum
        self._total_local_average += self._current_sum / self._current_n_instances
        self._total_n_instances += self._current_n_instances
        self._total_n_examples  += 1

        self._current_sum         = 0
        self._current_n_instances = 0

    def compute(self):
        return NestedAverage.NestedAverage(
            micro=self._total_sum / self._total_n_instances,
            macro=self._total_local_average / self._total_n_examples
        )
