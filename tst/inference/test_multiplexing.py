
def test_sharedVocabulary():
    """
    Testing if the "diamond inheritance" done in the multiplexer hierarchy actually just works.
    """
    from transformers import AutoTokenizer

    from tktkt.factories.preprocessing import IdentityPreprocessor
    from tktkt.wrappers.multiplexing import StochasticTokeniserMultiplexer_SameDomains, MultiplexedPreprocessor
    from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
    from tktkt.models.greedy.directional import L2R_Greedy

    core = AutoTokenizer.from_pretrained("roberta-base")

    tk_samedomain = StochasticTokeniserMultiplexer_SameDomains(preprocessor=MultiplexedPreprocessor(IdentityPreprocessor(), specific_preprocessors=False), subtokenisers=[
        HuggingFaceTokeniser(core),
        L2R_Greedy(preprocessor=IdentityPreprocessor(), vocab=core.get_vocab())
    ])

    for _ in range(20):
        tokens = tk_samedomain.prepareAndTokenise("expeditious")
        ids = [tk_samedomain.typeToId(t) for t in tokens]
        print(tokens)
        print(ids)
        print([tk_samedomain.idToType(i) for i in ids])
        print()


def test_differentVocabulary():
    """
    Toy example showing that two vocabularies, even when they have some overlapping types and many overlapping IDs,
    can be multiplexed by offsetting the IDs of the second, losslessly.
    """
    from tktkt.wrappers.multiplexing import StochasticTokeniserMultiplexer_DifferentDomains, MultiplexedPreprocessor
    from tktkt.models.word.segmentation import IdentityTokeniserWithVocab
    from tktkt.factories.preprocessing import TraditionalPreprocessor, IdentityPreprocessor

    tk1 = IdentityTokeniserWithVocab(IdentityPreprocessor(), {"a": 0, "b": 1, "c": 2, "d": 3, "[UNK]": 4}, unk_type="[UNK]")
    tk2 = IdentityTokeniserWithVocab(IdentityPreprocessor(), {"a": 0, "e": 1, "f": 2, "[UNK]": 3}, unk_type="[UNK]")
    tk = StochasticTokeniserMultiplexer_DifferentDomains(
        MultiplexedPreprocessor(
            TraditionalPreprocessor(),
            True
        ),
        [tk1, tk2]
    )

    for _ in range(10):
        tokens = tk.prepareAndTokenise("a b e &")
        ids = [tk.typeToId(t) for t in tokens]
        print(tokens)
        print(ids)
        print([tk.idToType(i) for i in ids])
        print()


if __name__ == "__main__":
    # test_sharedVocabulary()
    test_differentVocabulary()
