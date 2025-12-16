import logging
import datasets

from tktkt.models.bpe.vocabularisation import BPEVocabulariser, BpeTrainerImplementation
from tktkt.factories.preprocessors import ModernEnglishPreprocessor, KudoSpaceMarker
from tktkt.util.printing import setLoggingLevel

setLoggingLevel(logging.INFO)


def test_chizhov():
    vocabulariser = BPEVocabulariser(
        preprocessor=ModernEnglishPreprocessor(marker=KudoSpaceMarker),
        vocab_size=16,
        implementation=BpeTrainerImplementation.CHIZHOV
    )
    vocabulariser.vocabulariseFromStringIterable(["ab ab abab cd abc abcd xyzw"], name_if_not_named="test")


def test_chizhov_big():
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True).take(100_000)
    vocabulariser = BPEVocabulariser(
        preprocessor=ModernEnglishPreprocessor(marker=KudoSpaceMarker),
        vocab_size=32_000,
        implementation=BpeTrainerImplementation.CHIZHOV
    )
    vocabulariser.vocabulariseFromHf(dataset, text_field="text")


if __name__ == '__main__':
    test_chizhov()
    test_chizhov_big()
