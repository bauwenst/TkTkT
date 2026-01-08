# from .base import ClassicBPE
# from .vocabularisation import BPEVocabulariser, BpeTrainerImplementation, BPEArtifacts
# from .knockout import BPEKnockout, ReBPE, BPEKnockoutVocabulariser
# from .picky import PickyBPE, PickyBPEVocabulariser
# from .decomposing import TrimmedBPE, TrimmedBPEVocabulariser
# from .scaffold import ScaffoldBPE, ScaffoldBPEVocabulariser
#
# # Experimental
# from .dropout import BPEDropout, BPEBreakdown, BPEDropoutNonGeometric
# from .ensure import EnsuredBPE
# from .guided import GuidedBPEDropout
# from .shuffle import ShuffledBPE
# from .truncated import TruncatedBPE

###
# Although a nice way to gather tokeniser classes in one file, it is not wise to do the above if we expect people to
# use TkTkT in 3rd-party packages which will then get a submodule in TkTkT.
#
# For example, bpe_knockout.model.vocabulariser wants to import CacheableBPEArtifacts from tktkt.models.bpe.vocabularisation,
# which should work because the latter has nothing to do with bpe_knockout. Yet, we ALSO want to have a submodule
# tktkt.models.bpe.knockout which imports from bpe_knockout.model.vocabulariser. With the above imports uncommented,
# you will break e.g. all imports from bpe_knockout, because importing from tktkt.models.bpe.something first runs all
# the imports in tktkt.models.bpe, which you did not ask for. This part of Python sucks.
# You expect:
#     bpe_knockout.model.vocabulariser -> tktkt.models.bpe.vocabularisation
# but what happens is
#     bpe_knockout.model.vocabulariser -> tktkt.models.bpe -> tktkt.models.bpe.knockout -> bpe_knockout.model.vocabulariser
