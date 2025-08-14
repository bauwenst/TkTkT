def compareDeprecatedUnguided():
    from transformers import RobertaTokenizer

    from tktkt.models.predictive.viterbi import UnguidedViterbi
    from tktkt.models.predictive.viterbi.instances import LeastTokenViterbi
    from tktkt.preparation.instances import RobertaSpaceMarker, RobertaPreprocessor

    baseline = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    viterbi = UnguidedViterbi(baseline.get_vocab(), space_marker=RobertaSpaceMarker)
    viterbi2 = LeastTokenViterbi(RobertaPreprocessor, baseline.get_vocab(), max_step=None)
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

    from tktkt.models.predictive.viterbi.instances import ProductViterbi
    from tktkt.models.predictive.viterbi import RA_Product
    from tktkt.preparation.instances import RobertaPreprocessor

    from bpe_knockout.project.config import morphologyGenerator
    from transformers import RobertaTokenizer
    robbert_tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

    pre = RobertaPreprocessor
    vocab = robbert_tokenizer.get_vocab()

    tk1 = ProductViterbi(pre, vocab, max_step=None)
    tk2 = RA_Product(pre, vocab)

    for obj in morphologyGenerator():
        word = obj.word

        tokens1 = tk1.prepareAndTokenise(word)
        tokens2 = tk2.prepareAndTokenise(word)

        if tokens1 != tokens2:
            print(word)
            print("\t", tokens1)
            print("\t", tokens2)
