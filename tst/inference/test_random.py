from tktkt.models.random.randomfromvocab import RandomSegmentationFromVocab
from tktkt.preparation.instances import IdentityPreprocessor


def test_word():
    word = "reanimatie"
    vocab = {"re": 0, "anim": 1, "atie": 2, "reanim": 3, "at": 4, "ie": 5, "a": 6, "ean": 7, "r": 8}
    tk = RandomSegmentationFromVocab(IdentityPreprocessor, vocab)

    print([tk._segmentUsingIndices(word, seg) for seg in tk.generateSegmentations(word)])
    print(tk.tokenise(word))
    print(tk.tokenise(word))
    print(tk.tokenise(word))
    print(tk.tokenise(word))


if __name__ == "__main__":
    test_word()