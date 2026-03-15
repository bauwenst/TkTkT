from tst.preamble import *


def test_vocabulariser(N: int):
    import datasets

    from tktkt.models.ngram.vocabularisation import NgramVocabulariser
    from tktkt.factories.preprocessors import TraditionalPretokeniser, PretokeniserSequence, \
        AddWordBoundary, PrefixWhitespaceAsMarker, IdentityMapper, TruncateAndNormalise, Preprocessor

    preprocessor = Preprocessor(
        TruncateAndNormalise(1_000_000),
        IdentityMapper(),
        PretokeniserSequence([
            TraditionalPretokeniser,
            AddWordBoundary(PrefixWhitespaceAsMarker)
        ])
    )

    vc = NgramVocabulariser(preprocessor, N_min=N, N_max=N, truncate_to_top=64000, vocab_size=32678)
    vc.vocabulariseFromHf(datasets.load_dataset("allenai/c4", "en", streaming=True)["train"].take(3000), "text")


if __name__ == "__main__":
    test_vocabulariser(N=4)

