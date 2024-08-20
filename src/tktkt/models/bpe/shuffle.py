from typing import Set, List
import numpy.random as npr

from bpe_knockout.knockout.core import Merge
from .base import *

RNG = npr.default_rng(0)


class ShuffledBPE(DeterministicBPETokeniser):

    def __init__(self, preprocessor: Preprocessor, boundary_marker: BoundaryMarker,
                 vocab: Vocab, merges: MergeList, unk_type: str=None,

                 constrained: bool=True):
        super().__init__(
            preprocessor=preprocessor,
            boundary_marker=boundary_marker,
            unk_type=unk_type,

            vocab=vocab,
            merges=merges,

            do_morphemic_knockout=False
        )

        if constrained:
            self.shuffleMerges_constrained()
        else:
            self.shuffleMerges_naive()

    def shuffleMerges_naive(self):
        self._initialiseGraph(self.vocab, [" ".join(m.parts) for m in RNG.permutation(self.merge_graph.merges)])

    def shuffleMerges_constrained(self):
        """
        Randomly shuffles the BPE merges in such a way that the preorder specified by paths in the graph is preserved.
        That is: a merge that uses a type can never be shuffled to a position before the merge that builds that type.

        Note that preserving the preorder is a necessary but not sufficient condition to guarantee that a merge is
        applicable.
        """
        LOGGING_FRACTION = 0.1
        logging_step = int(LOGGING_FRACTION * len(self.merge_graph.merges))

        new_merges = []

        closed_set: Set[str]  = set(self.vocab) - {m.childType() for m in self.merge_graph.merges}  # Probably equal to {t for t,merges in self.merge_graph.merges_of.items() if merges}
        open_set: List[Merge] = [m for m in self.merge_graph.merges if all(part in closed_set for part in m.parts)]  # List so it can be sampled easier.

        while open_set:
            # Decide which merge comes next
            next_merge = open_set.pop(RNG.integers(len(open_set)))
            next_type  = next_merge.childType()
            new_merges.append(next_merge)
            closed_set.add(next_type)

            # Add newly unlocked merges to the open set. These will always use the new type.
            for m in self.merge_graph.merges_with[next_type]:
                if all(part in closed_set for part in m.parts):
                    open_set.append(m)

            if (len(new_merges)+1) % logging_step == 0:
                print(f"\tShuffled {len(new_merges)} merges...")

        self._initialiseGraph(vocab=self.vocab, mergelist=[" ".join(m.parts) for m in new_merges])

    def shuffleMerges_leafConstrained(self):
        """
        TODO: The most constrained version of BPE would be one where you try to preserve as many trees as possible.
              You could wonder what the point of that is, but it could be beneficial for unseen words.

        To do this, there are likely two approaches:
            1. Identify all leaves in the merge graph. Any transformation you make should let those leaves be tokenised
               into one token.
            2. Go through the merge list and colour every merge in the colour of a leaf tree it belongs to. Now you can
               use disentanglement algorithms like for register allocation by reasoning about these colours.
        """
        pass