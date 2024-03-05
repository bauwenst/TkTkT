from tst.preamble import *
from tst.evaluation.english_morphology import make_CanineViterbiBPE, TemporaryContext, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯

from tktkt.evaluation.morphological import tokeniseAndDecode, morphologyGenerator
from tktkt.visualisation.neural.splitpoints_probabilities import *

from fiject import MultiHistogram, CacheMode


def some_examples():
    # Classifier setup
    from tktkt.files.paths import from_pretrained_absolutePath, DataPaths
    from tktkt.models.viterbi.objectives_guided import HuggingFaceCharacterModelForTokenClassification, CanineTokenizer, CanineForTokenClassification
    tk = CanineTokenizer.from_pretrained("google/canine-c")
    core = from_pretrained_absolutePath(CanineForTokenClassification,
                                        DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28")
    classifier = HuggingFaceCharacterModelForTokenClassification(tk, core)

    words = [" establishmentarianism", " rainbow-coloured", " superbizarre", " algebraically", " ascertainably",
             " barelegged", " behaviourism", " chauvinistically", " maladministration",
             " ethnographically", " good-neighbourliness", " heavy-handedness",
             " existentialist", " imperialistically", " materialization", " decentralization", " disadvantageously",
             " nearsightedness", " neglectfulness"]
    for word in words:
        print(visualisePredictedBoundaries(classifier, word))


def celex_errors():
    # Easier way of getting the classifier, which is also set up for inputs of length < 4, unlike the above.
    canine_viterbi = make_CanineViterbiBPE()
    classifier = canine_viterbi.objectives[0].score_generator.nested_generator.logprob_classifier

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


def celex_probabilityDistribution():
    """
    Produces a histogram of all the predictions made by the character classifier.
    This way, we can visually verify whether most decisions are certain or whether most decisions are ambiguous.
    """
    canine_viterbi = make_CanineViterbiBPE()
    classifier = canine_viterbi.objectives[0].score_generator.nested_generator.logprob_classifier

    with TemporaryContext(setupEnglish()):
        histo = MultiHistogram("CANINE_boundary-probabilities_" + Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag(), caching=CacheMode.IF_MISSING)
        if histo.needs_computation:
            for obj in morphologyGenerator():
                histo.addMany("predictions", getPredictionProbabilities(classifier, obj.lemma()).tolist())

        histo.commit_histplot(binwidth=0.05, relative_counts=True, x_lims=(-0.025,1.025), x_tickspacing=0.1,
                              x_label="Predicted boundary probability", y_label="Proportion of words")


if __name__ == "__main__":
    # celex_errors()
    celex_probabilityDistribution()