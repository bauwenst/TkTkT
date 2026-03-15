from tktkt.evaluation.fertility import *
from tktkt.util.printing import dprint
from tktkt.factories.tokenisers import Factory_BPE, Factory_KudoPiece

from bpe_knockout import lexiconWeights

# TODO: The only actual fair comparison is BPE and KudoPiece trained on the exact same corpus with the exact same vocab size.
#       The most surprising result to me is that even though KudoPiece has 10k fewer subword types AND
#       it seems to allow fewer segmentations than BPE according to the below metric, it STILL outperforms
#       BPE when used for CANINE+Viterbi. The only possible explanation is that BPE has much, much, much
#       less meaningful subword types than KudoPiece: it allows many more segmentations and yet they're
#       mostly useless.
#       It does make some sense: BPE has to generate small-string trash to build subwords, whilst KudoPiece
#       can in theory have a disconnected set of morphemes, allowing minimal segmentations but much more
#       morphological matches.

from modest.languages.english import English_Celex


MAX = 17  # Well-under 1% of the dataset has length 17 (2^17 == 131k) yet it goes all the way to length 24 (2^24 = 16M).
# MAX = 24
DO_LOG = False

counts = lexiconWeights(override_reweighter=lambda x: x)
print("\nBPE")
dprint(getVocabStats(Factory_BPE().buildTokeniser(), raw_words=(o.word for o in English_Celex().generate()), counts=counts,
                     exclude_words_over_length=MAX, do_log_segmentations=DO_LOG).__dict__)
print("\nKudoPiece")
dprint(getVocabStats(Factory_KudoPiece().buildTokeniser(), raw_words=(o.word for o in English_Celex().generate()), counts=counts,
                     exclude_words_over_length=MAX, do_log_segmentations=DO_LOG).__dict__)
