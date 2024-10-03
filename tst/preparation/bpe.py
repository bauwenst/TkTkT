from tktkt.builders.english import getEnglishBpeFiles
from tktkt.preparation.instances import ModernEnglishPreprocessor, RobertaSpaceMarker
from tktkt.models.bpe.base import ClassicBPE


def test_preprocessAlreadySegmented():
    files = getEnglishBpeFiles()
    bpe = ClassicBPE(ModernEnglishPreprocessor(RobertaSpaceMarker), boundary_marker=RobertaSpaceMarker, vocab=files.loadVocabulary(), merges=files.loadMerges())

    input_string = "ab err ant"
    output_string = bpe._preprocessAlreadySegmentedString("ab err ant")
    print(output_string)
    assert len(output_string) == len(input_string) + 1


if __name__ == "__main__":
    test_preprocessAlreadySegmented()