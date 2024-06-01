from transformers import AutoTokenizer

from tktkt.evaluation.morphological import morphologyGenerator
from tktkt.models.bpe.base import ClassicBPE
from tktkt.models.bpe.guided import GuidedBPEDropout, ConstantCharacterClassifier


def test_classicbpe():
    """
    Test if TkTkT and HuggingFace produce the same tokens using their respective native BPE tokenisers.
    """
    roberta_hf = AutoTokenizer.from_pretrained("roberta-base")
    roberta_tktkt = ClassicBPE.fromHuggingFace(roberta_hf)  # Note: this is NOT a wrapper. It extracts vocab/merges and uses a custom BPE algorithm.

    from tktkt.evaluation.morphological import morphologyGenerator
    for obj in morphologyGenerator():
        word = obj.lemma()
        tokens1 = roberta_hf.tokenize(word)
        tokens2 = roberta_tktkt.prepareAndTokenise(word)
        assert tokens1 == tokens2, f"{word} becomes {tokens1} versus {tokens2}"


def test_guidedbpe():
    """
    Test if the implementation of guided BPE, which deviates from that of classic BPE, is equivalent to it if the
    dropout probability is 0.
    """
    base = ClassicBPE.fromHuggingFace(AutoTokenizer.from_pretrained("roberta-base"))
    guided_bpe = GuidedBPEDropout(
        base.preprocessor, base.merge_graph.vocab, [" ".join(m.parts) for m in base.merge_graph.merges], base.boundary_marker,
        dropout_probability=ConstantCharacterClassifier(p=0.0), always_dropout_above=None, unk_type=base.unk
    )

    for obj in morphologyGenerator():
        word = obj.lemma()
        tokens1 = base.prepareAndTokenise(word)
        tokens2 = guided_bpe.prepareAndTokenise(word)
        assert tokens1 == tokens2, f"{word} becomes {tokens1} versus {tokens2}"


if __name__ == "__main__":
    # test_classicbpe()
    test_guidedbpe()
