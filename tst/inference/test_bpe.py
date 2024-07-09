from transformers import AutoTokenizer

from tktkt.evaluation.morphological import morphologyGenerator
from tktkt.models.bpe.base import ClassicBPE
from tktkt.models.bpe.guided import GuidedBPEDropout, ConstantCharacterClassifier
from tktkt.models.bpe.ensure import EnsuredBPE
from tktkt.models.bpe.shuffle import ShuffledBPE
from tktkt.util.printing import lprint


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


def test_ensuredbpe():
    word = " antidisestablishmentarianism"

    classic_bpe: ClassicBPE = EnsuredBPE.fromHuggingFace(AutoTokenizer.from_pretrained("roberta-base"))
    print("Original vocab:", len(classic_bpe.vocab))
    print("Original tokenisation:", classic_bpe.prepareAndTokenise(word))
    print("Original last 7:")
    lprint(classic_bpe.merge_graph.merges[-7:], indent=1)

    ensured_bpe = EnsuredBPE(
        preprocessor=classic_bpe.preprocessor,
        boundary_marker=classic_bpe.boundary_marker,
        vocab=classic_bpe.vocab,
        merges=classic_bpe.merge_graph.getRawMerges(),

        ensure_strings=[word],
        forbid_strings=[],
        forbid_forming=[],
        do_preprocess_these=True,
        do_expand_vocabulary=False,
        do_binary_merges=True
    )
    print("New vocab:", len(ensured_bpe.vocab))
    print("New tokenisation:", ensured_bpe.prepareAndTokenise(word))
    print("New last 7:")
    lprint(ensured_bpe.merge_graph.merges[-7:], indent=1)


def test_shuffledbpe():
    example = " discombobulated"

    classic: ClassicBPE = ShuffledBPE.fromHuggingFace(AutoTokenizer.from_pretrained("roberta-base"))
    print("Old |V|:", len(classic.vocab))
    print("Old |M|:", len(classic.merge_graph.merges))
    print(classic.prepareAndTokenise(example))
    print()

    shuffled = ShuffledBPE(
        preprocessor=classic.preprocessor,
        boundary_marker=classic.boundary_marker,
        unk_type=classic.unk,

        vocab=classic.vocab,
        merges=classic.merge_graph.getRawMerges(),

        constrained=True
    )
    print("New |V|:", len(shuffled.vocab))
    print("New |M|:", len(shuffled.merge_graph.merges))
    print(shuffled.prepareAndTokenise(example))

    print("Old disabled:", len(classic.getDisabledTypes()))
    print("New disabled:", len(shuffled.getDisabledTypes()))

    print(classic.tokenise("Ġand"))
    print(shuffled.tokenise("Ġand"))


if __name__ == "__main__":
    # test_classicbpe()
    # test_guidedbpe()
    # test_ensuredbpe()
    test_shuffledbpe()