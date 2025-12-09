from transformers import AutoTokenizer

from tktkt.factories.tokenisers import Factory_BPE_Pythonic
from tktkt.factories.deserialisation import BPE50k_RobertaBase
from tktkt.models.bpe.guided import GuidedBPEDropout, ConstantCharacterClassifier
from tktkt.models.bpe.ensure import EnsuredBPE
from tktkt.models.bpe.shuffle import ShuffledBPE
from tktkt.util.printing import lprint

from modest.languages.english import English_Celex


def test_classicbpe():
    """
    Test if TkTkT and HuggingFace produce the same tokens using their respective native BPE tokenisers.
    """
    roberta_hf = AutoTokenizer.from_pretrained("roberta-base")
    roberta_tktkt = Factory_BPE_Pythonic(files=BPE50k_RobertaBase()).buildTokeniser()

    for obj in English_Celex().generate():
        word = obj.word
        tokens1 = roberta_hf.tokenize(word)
        tokens2 = roberta_tktkt.prepareAndTokenise(word)
        assert tokens1 == tokens2, f"{word} becomes {tokens1} versus {tokens2}"


def test_guidedbpe():
    """
    Test if the implementation of guided BPE, which deviates from that of classic BPE, is equivalent to it if the
    dropout probability is 0.
    """
    base = Factory_BPE_Pythonic(files=BPE50k_RobertaBase()).buildTokeniser()
    guided_bpe = GuidedBPEDropout(
        base.preprocessor, base.merge_graph.vocab, [" ".join(m.parts) for m in base.merge_graph.merges],
        dropout_probability=ConstantCharacterClassifier(p=0.0), always_dropout_above=None
    )

    for obj in English_Celex().generate():
        word = obj.word
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