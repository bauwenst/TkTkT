from tktkt.evaluation.fertility import *
from tktkt.util.printing import dprint

from tst.evaluation.english_morphology import make_English_BPE, make_English_KudoPiece
from bpe_knockout.project.config import morphologyGenerator, lexiconWeights, KnockoutDataConfiguration, setupEnglish

# TODO: The only actual fair comparison is BPE and KudoPiece trained on the exact same corpus with the exact same vocab size.
#       The most surprising result to me is that even though KudoPiece has 10k fewer subword types AND
#       it seems to allow fewer segmentations than BPE according to the below metric, it STILL outperforms
#       BPE when used for CANINE+Viterbi. The only possible explanation is that BPE has much, much, much
#       less meaningful subword types than KudoPiece: it allows many more segmentations and yet they're
#       mostly useless.
#       It does make some sense: BPE has to generate small-string trash to build subwords, whilst KudoPiece
#       can in theory have a disconnected set of morphemes, allowing minimal segmentations but much more
#       morphological matches.

MAX = 17  # Well-under 1% of the dataset has length 17 (2^17 == 131k) yet it goes all the way to length 24 (2^24 = 16M).
# MAX = 24
DO_LOG = False
with KnockoutDataConfiguration(setupEnglish()):
    counts = lexiconWeights(override_reweighter=lambda x: x)
    print("\nBPE")
    dprint(getVocabStats(make_English_BPE(), raw_words=(o.word for o in morphologyGenerator()), counts=counts,
                         exclude_words_over_length=MAX, do_log_segmentations=DO_LOG).__dict__)
    print("\nKudoPiece")
    dprint(getVocabStats(make_English_KudoPiece(), raw_words=(o.word for o in morphologyGenerator()), counts=counts,
                         exclude_words_over_length=MAX, do_log_segmentations=DO_LOG).__dict__)
