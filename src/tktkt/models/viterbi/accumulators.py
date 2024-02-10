from .lattice import ViterbiAccumulator


class Plus(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return previous_value + edge_score


class Max(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return max(previous_value, edge_score)


class Times(ViterbiAccumulator):

    def combine(self, previous_value: float, edge_score: float):
        return previous_value * edge_score
