def compareDeprecatedUnguided():
    from transformers import RobertaTokenizer

    from tktkt.models.viterbi._deprecated import UnguidedViterbi
    from tktkt.models.viterbi.instances import LeastTokenViterbi
    from tktkt.preparation.splitters import RobertaPretokeniser
    from tktkt.preparation.spacemarking import ROBERTA_SPACING

    baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    viterbi = UnguidedViterbi(baseline.get_vocab(), space_marker=ROBERTA_SPACING)
    viterbi2 = LeastTokenViterbi(RobertaPretokeniser, baseline.get_vocab(), max_step=None)
    examples = [" coronamaatregelen", " tyrannosaurus", " departementspolitiek", " acteursloopbaan",
                " gewichtheffen", " schoonheidsideaal", " softwarepakket", " gekkenwerk",
                " relatietherapie", " medialeugens", " pianosolo", " flatscreentelevisie", " boseend",
                " palmeiland", " aswoensdag", " operatiekwartier", " rekentijd"]
    for e in examples:
        print("RobBERT:", baseline.tokenize(e))
        print("Viterbi_dep:", viterbi.prepareAndTokenise(e))
        print("Viterbi_new:", viterbi2.prepareAndTokenise(e))
        print()


def compareDeprectatedProduct():

    from tktkt.models.viterbi.instances import ProductViterbi
    from tktkt.models.viterbi._deprecated import RA_Product
    from tktkt.preparation.splitters import RobertaPretokeniser

    from bpe_knockout.project.config import morphologyGenerator
    from bpe_knockout.auxiliary.robbert_tokenizer import robbert_tokenizer

    pre = RobertaPretokeniser
    vocab = robbert_tokenizer.get_vocab()

    tk1 = ProductViterbi(pre, vocab, max_step=None)
    tk2 = RA_Product(pre, vocab)

    for obj in morphologyGenerator():
        word = obj.lemma()

        tokens1 = tk1.prepareAndTokenise(word)
        tokens2 = tk2.prepareAndTokenise(word)

        if tokens1 != tokens2:
            print(word)
            print("\t", tokens1)
            print("\t", tokens2)

