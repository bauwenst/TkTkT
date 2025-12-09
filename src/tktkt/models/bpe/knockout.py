from typing import Optional

from bpe_knockout import ReferenceMode
from bpe_knockout.model.auto import AutoKnockout, BTEConfig, KnockoutConfig, AnnealingConfig, ReifyMode
from modest.interfaces.datasets import ModestDataset

from .base import _DeterministicBPETokeniser, MergeList
from ...interfaces.tokeniser import *


class _KnockoutJIT(_DeterministicBPETokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList,
                 reference_segmentations: Optional[ModestDataset],
                 iterations: int, do_knockout: bool, do_reify: bool, backwards_compatible: bool=False):
        if reference_segmentations is None:
            super().__init__(preprocessor=preprocessor, vocab=vocab, merges=merges)
        else:  # Make a config and run AutoKnockout, which uses the given dataset to modify a BTE tokeniser during the same runtime as its tokeniser is instantiated.
            do_anneal = False
            config = BTEConfig(
                knockout=KnockoutConfig(reference=ReferenceMode.NONE if not do_knockout else ReferenceMode.MORPHEMIC),
                annealing=AnnealingConfig(reference=ReferenceMode.NONE if not do_anneal else ReferenceMode.MORPHEMIC),
                reify=ReifyMode.NONE if not do_reify else ReifyMode.FIX_AND_LINK if backwards_compatible else ReifyMode.FIX_AND_LINK_AND_MAKE,
                iterations=iterations,
            )
            bte = AutoKnockout(config).from_objects(preprocessor=preprocessor, vocab=vocab, merges=merges, reference=reference_segmentations)
            super().__init__(preprocessor=preprocessor, vocab=bte.vocab, merges=bte.merge_graph.getRawMerges(), metadata=config)


class BPEKnockout(_KnockoutJIT):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList,
                 reference_segmentations: Optional[ModestDataset]):
        super().__init__(
            preprocessor=preprocessor,
            vocab=vocab,
            merges=merges,

            reference_segmentations=reference_segmentations,

            do_knockout=True,
            do_reify=False,
            iterations=1
        )


class ReBPE(_KnockoutJIT):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab, merges: MergeList,
                 reference_segmentations: Optional[ModestDataset],
                 iterations: int, backwards_compatible: bool=False):
        super().__init__(
            preprocessor=preprocessor,
            vocab=vocab,
            merges=merges,

            reference_segmentations=reference_segmentations,

            do_knockout=True,
            do_reify=True,
            iterations=iterations,
            backwards_compatible=backwards_compatible
        )
