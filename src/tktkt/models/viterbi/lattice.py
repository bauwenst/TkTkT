from dataclasses import dataclass
from typing import Callable


INFTY = float("inf")

class ViterbiNode:
    def __init__(self, id=None):
        self.id = id
        self.current_loss = INFTY
        self.backpointer  = None

    def __repr__(self):
        return f"N({self.id} -> {self.backpointer.id if self.backpointer else None})"


@dataclass
class ViterbiLoss:
    loss: float
    tiebreaker: float

    def __lt__(self, other):
        return (self.loss, self.tiebreaker) < (other.loss, other.tiebreaker)

@dataclass
class ViterbiLossUpdater:  # Doesn't use abstract methods because the implementation of the two methods is independent.
    loss_update: Callable[[ViterbiLoss, ...], float]
    tiebreaker_update: Callable[[ViterbiLoss, ...], float]

    def update(self, previous_loss: ViterbiLoss, *args, **kwargs) -> ViterbiLoss:
        return ViterbiLoss(self.loss_update(previous_loss, *args, **kwargs),
                           self.tiebreaker_update(previous_loss, *args, **kwargs))
