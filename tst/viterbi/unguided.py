# TODO: Implement RA_Product in this framework and compare it to our deprecated implementation.

from transformers import RobertaTokenizer

from src.tktkt.models.viterbi._deprecated import UnguidedViterbi
from src.tktkt.preparation.spacemarking import ROBERTA_SPACING

from tst.viterbi.framework import makeTokeniser

baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
viterbi = UnguidedViterbi(baseline.get_vocab(), space_marker=ROBERTA_SPACING)
viterbi2 = makeTokeniser()
examples = [" coronamaatregelen", " tyrannosaurus", " departementspolitiek", " acteursloopbaan",
            " gewichtheffen", " schoonheidsideaal", " softwarepakket", " gekkenwerk",
            " relatietherapie", " medialeugens", " pianosolo", " flatscreentelevisie", " boseend",
            " palmeiland", " aswoensdag", " operatiekwartier", " rekentijd"]
for e in examples:
    print("RobBERT:", baseline.tokenize(e))
    print("Viterbi_old:", viterbi.prepareAndTokenise(e))
    print("Viterbi_new:", viterbi2.prepareAndTokenise(e))
    print()