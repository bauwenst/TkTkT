from tktkt.models.compressive.lzw import LzwVocabulariser
from tktkt.factories.preprocessing import ModernEnglishPreprocessor, KudoSpaceMarker


def fromSingleSentence():
    preprocessor = ModernEnglishPreprocessor(KudoSpaceMarker)
    vocabulariser = LzwVocabulariser(preprocessor, vocab_size=290)

    from tktkt.util.types import NamedIterable
    sentence = "there is a house in your backyard -- a treehouse you might say"
    vocabulariser.vocabulariseFromStringIterable(NamedIterable([sentence], "test"))

    vocabulariser.vocab_size = 1_000_000
    vocabulariser.vocabulariseFromStringIterable(NamedIterable([sentence], "test"))


def fromDataset():
    preprocessor = ModernEnglishPreprocessor(KudoSpaceMarker)
    vocabulariser = LzwVocabulariser(preprocessor, vocab_size=32768)

    from datasets import load_dataset
    vocabulariser.vocabulariseFromHf(load_dataset("allenai/c4", "en", streaming=True)["train"].take(10_000), text_field="text")


if __name__ == "__main__":
    fromDataset()
