from tktkt.factories.deserialisation import BPE32ki_SlimPajama3M
from tktkt.models.bpe.base import ClassicBPE


def test_preprocessAlreadySegmented():
    files = BPE32ki_SlimPajama3M()
    bpe = ClassicBPE(files.preprocessorEffective(), vocab=files.buildVocabulary(), merges=files.buildMerges())

    input_string = "ab err ant"
    output_string = bpe._preprocessAlreadySegmentedString("ab err ant")
    print(output_string)
    assert len(output_string) == len(input_string) + 1


if __name__ == "__main__":
    test_preprocessAlreadySegmented()