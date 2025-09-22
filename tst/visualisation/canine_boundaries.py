from tktkt.models.random.generationbased import generateSegmentationIndices_fixedSpace
from tktkt.util.strings import indicesToTokens
from tst.evaluation.english_morphology import KnockoutDataConfiguration, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯

from tktkt.factories.tokenisers import Factory_BoMMaSum_BPE
from tktkt.evaluation.morphological import morphologyGenerator
from tktkt.interfaces.tokeniser import prepare_tokenise_decode
from tktkt.visualisation.neural.splitpoints_probabilities import *

from fiject import MultiHistogram, CacheMode


def test_probabilityVisualisation():
    # Classifier setup (this is only for illustration purposes; normally you would use a Factory for this!).
    from tktkt.paths import from_pretrained_absolutePath, TkTkTPaths
    from tktkt.models.predictive.viterbi.objectives_guided import HuggingFaceForBinaryCharacterClassification, CanineTokenizer, CanineForTokenClassification
    tk = CanineTokenizer.from_pretrained("google/canine-c")
    core = from_pretrained_absolutePath(CanineForTokenClassification,
                                        TkTkTPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28")
    classifier = HuggingFaceForBinaryCharacterClassification(tk, core)

    # Visualise the following words
    words = [" establishmentarianism", " rainbow-coloured", " superbizarre", " algebraically", " ascertainably",
             " barelegged", " behaviourism", " chauvinistically", " maladministration",
             " ethnographically", " good-neighbourliness", " heavy-handedness",
             " existentialist", " imperialistically", " materialization", " decentralization", " disadvantageously",
             " nearsightedness", " neglectfulness"]
    for word in words:
        print(visualisePredictedBoundaries(classifier, word))


def test_visualiseCelexMismatches():
    """
    Print Viterbi predictions that don't match CELEX boundaries.
    """
    canine_viterbi = Factory_BoMMaSum_BPE().buildTokeniser()  # Easier way of getting the classifier, which is also set up for inputs of length < 4, unlike the above.
    classifier = canine_viterbi.objectives[0].score_generator.nested_generator.logprob_classifier
    vocab = canine_viterbi.objectives[0].score_generator.vocab

    with KnockoutDataConfiguration(setupEnglish()):
        for obj in morphologyGenerator(verbose=False):
            word = obj.word

            reference = " ".join(obj.segment())
            viterbi   = " ".join(prepare_tokenise_decode(word, canine_viterbi, canine_viterbi.preprocessor)).strip()

            if reference != viterbi:
                print(word)
                word = canine_viterbi.preprocessor.do(word)[0]
                print("\tGold reference:  ", reference)
                print("\tProbabilities:   ", visualisePredictedBoundaries(classifier, word))
                print("\tViterbi decision:", viterbi)
                print("\t            `--->", " // ".join(
                    map(lambda tokens: " ".join(filter(lambda t: t, tokens)),
                        map(canine_viterbi.preprocessor.undo_per_token,
                            sorted(
                                filter(lambda segmentation: not any(len(t) == 1 for t in segmentation[1:]),
                                    map(lambda idcs: indicesToTokens(word, idcs),
                                        generateSegmentationIndices_fixedSpace(word, vocab)
                                        )
                                ),
                                reverse=True
                            )
                        )
                    )
                ))


def test_uncertaintyOfPredictions():
    """
    Produces a histogram of all the predictions made by the character classifier.
    This way, we can visually verify whether most decisions are certain or whether most decisions are ambiguous.
    """
    with KnockoutDataConfiguration(setupEnglish()):
        histo = MultiHistogram("CANINE_boundary-probabilities_" + Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag(), caching=CacheMode.IF_MISSING)

        if histo.needs_computation:
            canine_viterbi = Factory_BoMMaSum_BPE().buildTokeniser()
            classifier = canine_viterbi.objectives[0].score_generator.nested_generator.logprob_classifier

            for obj in morphologyGenerator():
                histo.addMany("predictions", getPredictionProbabilities(classifier, obj.word).tolist())

        histo.commit_histplot(binwidth=0.05, relative_counts=True, x_lims=(-0.025,1.025), x_tickspacing=0.1,
                              x_label="Predicted boundary probability", y_label="Proportion of words")


if __name__ == "__main__":
    # test_visualiseCelexMismatches()
    test_uncertaintyOfPredictions()