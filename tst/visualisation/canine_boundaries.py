from tktkt.evaluation.morphological import tokeniseAndDecode, morphologyGenerator
from tst.evaluation.english_morphology import make_CanineViterbiBPE, TemporaryContext, setupEnglish

from tktkt.visualisation.neural.splitpoints_probabilities import *

# Path setup
from tktkt.files.paths import from_pretrained_absolutePath, DataPaths
from tst.preamble import *

canine_viterbi = make_CanineViterbiBPE()

# Classifier setup
from tktkt.models.viterbi.objectives_guided import HuggingFaceCharacterModelForTokenClassification, CanineTokenizer, CanineForTokenClassification
tk   = CanineTokenizer.from_pretrained("google/canine-c")
core = from_pretrained_absolutePath(CanineForTokenClassification, DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28")
classifier = HuggingFaceCharacterModelForTokenClassification(tk, core)

def some_examples():
    words = [" establishmentarianism", " rainbow-coloured", " superbizarre", " algebraically", " ascertainably",
             " barelegged", " behaviourism", " chauvinistically", " maladministration",
             " ethnographically", " good-neighbourliness", " heavy-handedness",
             " existentialist", " imperialistically", " materialization", " decentralization", " disadvantageously",
             " nearsightedness", " neglectfulness"]
    for word in words:
        print(visualisePredictedBoundaries(classifier, word))


def celex_errors():
    classifier = canine_viterbi.objectives[0].score_generator.nested_generator.logprob_classifier  # Is set up for small inputs, unlike the above classifier.

    with TemporaryContext(setupEnglish()):
        for obj in morphologyGenerator(verbose=False):
            word = obj.lemma()

            reference = obj.morphSplit()
            viterbi   = " ".join(tokeniseAndDecode(word, canine_viterbi)).strip()

            if reference != viterbi:
                print(word)
                print("\tGold reference:  ", reference)
                print("\tProbabilities:   ", visualisePredictedBoundaries(classifier, word))
                print("\tViterbi decision:", viterbi)
                print("\t            `--->", " // ".join(
                    map(lambda tokens: " ".join(filter(lambda t: t, tokens)),
                        map(canine_viterbi.preprocessor.undo_per_token,
                            sorted(
                                filter(lambda segmentation: not any(len(t) == 1 for t in segmentation[1:]),
                                    canine_viterbi.objectives[0].score_generator.getAllPossibleSegmentations(
                                        canine_viterbi.preprocessor.do(word)[0], max_k=20
                                    )
                                ),
                                reverse=True
                            )
                        )
                    )
                ))


if __name__ == "__main__":
    celex_errors()
