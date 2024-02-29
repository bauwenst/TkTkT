from transformers import RobertaTokenizer

from src.tktkt.preparation.splitters import RobertaPretokeniser
from src.tktkt.models.viterbi.objectives_guided import *
from src.tktkt.models.viterbi.instances import HFPointViterbi

baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
vocab = baseline.get_vocab()

checkpoint = "google/canine-c"  # By using a checkpoint that wasn't trained on tokenisation, you'll get random boundary probabilities, so all this doesn't say much.


tk = HFPointViterbi(
    RobertaPretokeniser,
    vocab,
    15,
    checkpoint,
    CanineTokenizer,
    CanineForTokenClassification
)

words = [" flatscreentelevisie"]
for word in words:
    print(word)
    print("\tRobBERT BPE:", baseline.tokenize(word))
    print("\tSame vocab, new inference:", tk.prepareAndTokenise(word))
