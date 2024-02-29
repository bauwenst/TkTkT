from .framework import ViterbiAccumulator


class ScoreSum(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return previous_value + edge_score


class ScoreMax(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return max(previous_value, edge_score)


class ScoreProduct(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return previous_value * edge_score


class ScoreSubtract(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return previous_value - edge_score