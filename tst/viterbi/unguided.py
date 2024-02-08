from transformers import RobertaTokenizer

from src.tktkt.models.viterbi.unguided import UnguidedViterbi
from src.tktkt.preparation.spacemarking import ROBERTA_SPACING

baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
viterbi = UnguidedViterbi(baseline.get_vocab(), space_marker=ROBERTA_SPACING)
examples = [" coronamaatregelen", " tyrannosaurus", " departementspolitiek", " acteursloopbaan",
            " gewichtheffen", " schoonheidsideaal", " softwarepakket", " gekkenwerk",
            " relatietherapie", " medialeugens", " pianosolo", " flatscreentelevisie", " boseend",
            " palmeiland", " aswoensdag", " operatiekwartier", " rekentijd"]
for e in examples:
    print("RobBERT:", baseline.tokenize(e))
    print("Viterbi:", viterbi.prepareAndTokenise(e))
    print()