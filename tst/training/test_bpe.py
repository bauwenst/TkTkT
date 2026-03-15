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


def test_picky():
    from tktkt.util.types import pipe, dictget
    corpus = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True).filter(pipe(pipe(dictget("text"), str.strip), bool)).take(50_000)

    from tktkt.factories.preprocessors import Prefab2, KudoSpaceMarker
    from tktkt.models.bpe.picky import PickyBPEVocabulariser
    vocabulariser = PickyBPEVocabulariser(
        preprocessor=Prefab2(KudoSpaceMarker),
        vocab_size=32768,
        picky_threshold=0.5,
        character_coverage=0.9999,
        max_type_length=16
    )
    vocabulariser.vocabulariseFromHf(corpus, text_field="text")


if __name__ == '__main__':
    test_chizhov()
    test_chizhov_big()
    test_picky()
