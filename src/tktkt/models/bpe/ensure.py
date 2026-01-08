"""
EnsuredBPE is a modification to an existing BPE tokeniser which ensures that a given set of (sub)words appear in the
vocabulary in their entirety.

TODO: What isn't possible currently is path insurance, where you can ensure that a merge a+b->ab is added first
      and then expect a string a b c d to (somehow) become abcd VIA that merge a+b, because the merge b+c->bc could be
      in the tokeniser already. The reason that would be powerful is that you could e.g. force compounds to be merged from constituents.
"""
import warnings
from typing import Iterable

from bpe_knockout.model.vocabulariser import BPEKnockoutVocabulariser
from bpe_knockout.model.graph import Merge

from .base import _DeterministicBPETokeniser, MergeList
from ...interfaces.tokenisers import *
from ...util.iterables import deduplicate, mapExtend
from ...util.functions import relu


class EnsuredBPE(_DeterministicBPETokeniser[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor,
                 vocab: Vocab[WithSpecials], merges: MergeList, ensure_strings: Iterable[str], forbid_strings: Iterable[str], forbid_forming: Iterable[str],
                 do_preprocess_these: bool=False, do_expand_vocabulary: bool=False, do_binary_merges: bool=True):
        """
        :param ensure_strings: The (sub)words for which there must be a single token in the BPE vocabulary (that can be
                               formed). They are added in the given order, which can affect the resulting learnt merges.
        :param forbid_strings: The (sub)words for which there cannot be a single token in the BPE vocabulary. These are
                               removed using BPE-knockout's knockout procedure, after ensuring all the above strings.
        :param forbid_forming: The (sub)words which can stay in the vocabulary, but cannot be formed by any merge tree
                               (e.g. [UNK]).

        :param do_preprocess_these: Whether the given (sub)words need to still be preprocessed by the given preprocessor.
                                    If split into multiple pretokens, they are ensured separately.
        :param do_expand_vocabulary: Whether to exceed the current size of the vocabulary. If false, the last types in
                                     the existing tokeniser will be removed.
        """
        super().__init__(preprocessor=preprocessor, vocab=vocab, merges=merges)

        self.ensured   = list(deduplicate(mapExtend(self.preprocessor.do, ensure_strings) if do_preprocess_these else ensure_strings))
        self.forbidden = list(deduplicate(mapExtend(self.preprocessor.do, forbid_strings) if do_preprocess_these else forbid_strings))
        self.special   = list(deduplicate(mapExtend(self.preprocessor.do, forbid_forming) if do_preprocess_these else forbid_forming))

        self._binary     = do_binary_merges
        self._fixed_size = not do_expand_vocabulary

        self._trained = False
        self._train()

    def _train(self):
        if self._trained:
            raise RuntimeError("Cannot train EnsuredBPE twice.")

        # First, let's find which merges we can delete to make room for the ensured strings' merges.
        # (You have to do that now, because we're soon going to be adding merges we don't want to delete.)
        deletable_merge_stack: list[Merge] = []
        if self._fixed_size:
            # Find all merges that need to be protected against trimming from the end.
            protected_merge_priorities = set()
            for ensured_string in self.ensured:
                _, index_to_priority = BPEKnockoutVocabulariser._tokenise_diagnostic(self, ensured_string)
                protected_merge_priorities.update(index_to_priority.values())

            deletable_merge_stack = sorted(filter(lambda m: m.priority not in protected_merge_priorities, self.merge_graph.merges))

        ### PART 1: Ensure strings ###
        total_merges_added = 0
        for ensured_string in self.ensured:
            print("Ensuring", ensured_string, "...")

            tokens = self.tokenise(ensured_string)
            while len(tokens) > 1:
                # print(tokens)
                best_merge: tuple = None
                if not self._binary:
                    best_merge = tuple(tokens)
                    tokens = ["".join(tokens)]
                else:
                    # TODO: Ordering merges should technically be done by segmenting a corpus and counting the possible
                    #       pairs of adjacent tokens we have to choose between here.
                    #       It takes about 30 minutes to tokenise 5 million words with the native TkTkT BPE tokeniser, which
                    #       is quite a lot of work. Hence, we use two trivially faster (but technically incorrect) alternatives:
                    #           - Priority-based (current): do the merge first where the sum of the priorities of the merges that make the parts is lowest.
                    #           - FIFO: merge token 1 and 2, token 3 and 4, token 5 and 6, ...
                    lowest_combined_priority = +float("inf")
                    for left, right in zip(tokens[:-1], tokens[1:]):
                        merges_that_form_left  = self.merge_graph.merges_of[left]   # <--- This line is what requires you to immediately add merges every iteration; you need to get the merge that forms types you just added.
                        merges_that_form_right = self.merge_graph.merges_of[right]

                        lowest_left_priority  = 0 if not merges_that_form_left  else min(merges_that_form_left).priority
                        lowest_right_priority = 0 if not merges_that_form_right else min(merges_that_form_right).priority

                        combined_priority = lowest_left_priority + lowest_right_priority
                        if combined_priority < lowest_combined_priority:
                            lowest_combined_priority = combined_priority
                            best_merge = (left, right)

                    assert best_merge is not None

                    # Apply the merge to the current token buffer.
                    # The reason we don't do tokens = self.tokenise(ensured_string) every iteration is that we want to
                    # save as many self.syncWithGraph() calls as possible, which is the actually expensive part of
                    # modifying the BPE graph. Hence, the graph inside the tokeniser is up-to-date, but the tokeniser's cache is out-of-date.
                    new_tokens = []
                    i = 0
                    while i < len(tokens):
                        if i == len(tokens)-1 or not(tokens[i] == best_merge[0] and tokens[i+1] == best_merge[1]):
                            new_tokens.append(tokens[i])
                            i += 1
                        else:
                            new_tokens.append(tokens[i] + tokens[i+1])
                            i += 2
                    tokens = new_tokens

                # Update the graph, but not the tokeniser.
                # Note that merge priorities will be as late as possible, which makes sense since these are the latest merges you should do.
                self.merge_graph.addArc(" ".join(best_merge))
                total_merges_added += 1

        # Now that you're going to call .tokenise() in the next iteration, synchronise the tokeniser with the new knowledge.
        self._syncWithGraph()

        # Finally, prune away as many types as you added (minus the amount you know you're already going to prune anyway).
        if self._fixed_size:
            total_merges_to_remove = relu(total_merges_added - len(self.forbidden))  # Will be 0 when e.g. you add no merges, or you know in advance you're going to delete a tonne already.
            if len(deletable_merge_stack) < total_merges_to_remove:
                warnings.warn(f"Vocabulary will be expanded beyond its original size, because there are no old merges left to remove and yet we still need to remove {total_merges_to_remove - len(deletable_merge_stack)}...")

            if total_merges_to_remove:
                deletable_merge_stack = deletable_merge_stack[-total_merges_to_remove:]  # Note that L[-0:] is the whole list, hence the extra `if`.
                for m in reversed(deletable_merge_stack):
                    self.merge_graph.knockout(m.childType())

        ### PART 2: Forbid strings ###
        for forbidden in self.forbidden:
            if forbidden in self.vocab:
                self.merge_graph.knockout(type_to_delete=forbidden)
        self._syncWithGraph()

        ### PART 3: Special strings ###
        for special in self.special:
            if special in self.vocab:
                self.merge_graph.knockout(type_to_delete=special)
            self.merge_graph.addVertex(type_to_add=special)
        self._syncWithGraph()

        self._trained = True
