from tktkt.models.random.randomfromvocab import RandomSegmentationFromVocab, segmentUsingIndices, generateSegmentationIndices_exponentialSpace
from tktkt.preparation.instances import IdentityPreprocessor


def test_word():
    word = "reanimatie"
    vocab = {"re": 0, "anim": 1, "atie": 2, "reanim": 3, "at": 4, "ie": 5, "a": 6, "ean": 7, "r": 8}
    tk = RandomSegmentationFromVocab(IdentityPreprocessor, vocab)

    print([segmentUsingIndices(word, seg) for seg in generateSegmentationIndices_exponentialSpace(word, vocab)])
    print(tk.tokenise(word))
    print(tk.tokenise(word))
    print(tk.tokenise(word))
    print(tk.tokenise(word))


def test_segmentationAmount():
    from transformers import AutoTokenizer
    from tktkt.models.bpe.base import ClassicBPE
    from tktkt.evaluation.fertility import possibleSegmentations
    from bpe_knockout.project.config import KnockoutDataConfiguration, setupDutch, morphologyGenerator

    tk = ClassicBPE.fromHuggingFace(AutoTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base"))
    with KnockoutDataConfiguration(setupDutch()):
        for obj in morphologyGenerator():
            word = obj.word
            print(word, len(word))
            assert len(generateSegmentationIndices_exponentialSpace(word, tk.vocab)) == possibleSegmentations(tk.vocab, word)


if __name__ == "__main__":
    test_segmentationAmount()