from tktkt.evaluation.morphological import tokeniseAndDecode, morphologyGenerator
from tst.evaluation.english_morphology import canine_viterbi, TemporaryContext, setupEnglish

from tktkt.visualisation.neural.splitpoints_probabilities import *

# Path setup
from tktkt.files.paths import from_pretrained_absolutePath
from tst.preamble import *

# Classifier setup
from tktkt.models.viterbi.objectives_guided import HuggingFaceCharacterModelForTokenClassification, CanineTokenizer, CanineForTokenClassification
tk   = CanineTokenizer.from_pretrained("google/canine-c")
core = from_pretrained_absolutePath(CanineForTokenClassification, checkpoints_path / "CANINE-C_2024-02-12_19-35-28")
classifier = HuggingFaceCharacterModelForTokenClassification(tk, core)

def some_examples():
    words = [" establishmentarianism", " rainbow-coloured", " superbizarre", " algebraically", " ascertainably",
             " barelegged", " behaviourism", " chauvinistically", " maladministration",
             " ethnographically", " good-neighbourliness", " heavy-handedness",
             " existentialist", " imperialistically", " materialization", " decentralization", " disadvantageously",
             " nearsightedness", " neglectfulness"]
    for word in words:
        print(visualisePredictedBoundaries(classifier, word))


def celex():
    classifier = canine_viterbi.objectives[0].score_generator.nested_generator.model  # Is set up for small inputs, unlike the above classifier.

    with TemporaryContext(setupEnglish()):
        for obj in morphologyGenerator(verbose=False):
            word = obj.lemma()
            print(word)
            print("\t", visualisePredictedBoundaries(classifier, word))             # What the model predicts
            print("\t", " ".join(tokeniseAndDecode(word, canine_viterbi)).strip())  # What Viterbi makes of it


if __name__ == "__main__":
    celex()
