"""
Disables merges, rather than undoing them afterwards.
"""
from typing import Iterable

from bpe_knockout.model.graph import MergeGraph

from .base import _DeterministicBPETokeniser, MergeList
from ...interfaces.tokenisers import *


class TruncatedBPE(_DeterministicBPETokeniser[WithSpecials]):
    """
    Limits BPE to an earlier part of training by truncating the merge list.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], merges: MergeList, delta_vocab_size: int):
        super().__init__(preprocessor=preprocessor, vocab=vocab, merges=merges)
        deleted_types = list(merge.childType() for merge in self.merge_graph.merges[-delta_vocab_size:])
        for typ in reversed(deleted_types):
            self.merge_graph.detach(typ, cascade=False)  # Cascading will have no effect anyway since it's the end of the vocab.
        self._syncWithGraph()  # Updates merge cache and settles vocabulary specials.


class SubBPE(_DeterministicBPETokeniser[WithSpecials]):
    """
    Rather than just truncating the merge list, this one deletes specific types and the cascade of all types that depend on them.
    Also works for algorithms like VOLT that start with an existing tokeniser's vocabulary and then select the most relevant types from it.
    """

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials], merges: MergeList,
                 whitelist: Optional[Iterable[str]]=None, blacklist: Optional[Iterable[str]]=None, invert_whitelist: bool=False):
        """
        :param whitelist: Types to KEEP AT LEAST, even if that means keeping extra ancestor types to be able to build them with BPE.
        :param blacklist: Types to DELETE AT LEAST, even if that means deleting extra descendant types that will no longer be formed with BPE.
        :param invert_whitelist: If a whitelist is given, this option will treat the input as if blacklist = vocab - whitelist was given,
                                 which may result in the deletion of types that are in the whitelist.
        """
        whitelist = set(whitelist or [])
        blacklist = set(blacklist or [])
        if not whitelist and not blacklist:
            raise ValueError("A sub-tokeniser of a BPE tokeniser is specified by the types to conserve or the types to delete.")
        elif whitelist and blacklist:
            raise ValueError("Specified a whitelist and a blacklist simultaneously.")
        elif invert_whitelist and not whitelist:
            raise ValueError("To invert the whitelist, one must be given.")

        super().__init__(preprocessor=preprocessor, vocab=vocab, merges=merges)

        # We implement both cases using a blacklist. So, if there is a whitelist, it has to be converted first.
        if whitelist:
            blacklist = set(vocab) - whitelist  # These are the types the user does not want...
            if not invert_whitelist:  # ...unless those types make things in the whitelist.
                whitelist_ancestors = set()
                for typ in whitelist:
                    whitelist_ancestors.update(self.merge_graph.getAncestors(typ))
                blacklist -= whitelist_ancestors

        # Use the blacklist to cut types out of the BPE graph.
        while blacklist:
            newly_deleted_types = self.merge_graph.detach(blacklist.pop(), cascade=True)
            blacklist -= newly_deleted_types
        self._syncWithGraph()
