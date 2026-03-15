"""
Evaluate any tokeniser on English morphology.

TODO: There are two issues with our CANINE evaluation.
    1. I'm not sure if it got special tokens during pretraining, and it is likely not a good idea to leave them out in
       both fine-tuning and inference. The model is used to using these as working memory, most likely.
        - In fine-tuning, you would need to pad the labels based on special tokens. There has to be a function for this
          in HuggingFace because which special tokens are added is a tokeniser-specific decision that can't be predicted.
        - In inference, you would need to filter these from the prediction output before handing them to your Viterbi lattice.
    2. Because the vocabulary I am using to limit the Viterbi steps during inference is specified in pseudo-bytes, I am
       giving CANINE an input in pseudo-bytes for inference too. This is a problem, because CANINE wasn't pre-trained
       nor fine-tuned with pseudo-bytes, but with a modulo mapping.
       What this means is that ë is going to show up as Ã<< and CANINE won't know what it means since it only ever saw
       ë during training and fine-tuning.
       |
       What you have here is a problem caused by the fact that we are using a language model as tokeniser for another
       language model with different pretokenisation. Indeed, we need two pretokenisers, and one of them should be
       applied AFTER the tokeniser!
         1. Give the original text to CANINE. (It uses context to figure out where to put segmentation boundaries.)
         2. Invert the pseudo-byte mapping of the vocabulary to recognise which steps correspond to valid tokens, and give the
            result to the Viterbi tokeniser. Now you have segmentations into strings that include spaces and ë etc.
         2. Apply the byte mapping of the LM to map these tokens into the LM vocabulary.
"""
import json

from tktkt.factories.tokenisers import *
from tktkt.util.timing import datetimeDashed
from tktkt.evaluation.morphological import intrinsicEvaluation
from tktkt.models.predictive.viterbi import *
from tktkt.paths import TkTkTPaths

from modest.languages.english import English_Celex


def evaluateTokenisers(tokenisers: Iterable[Tokeniser]):
    dataset = English_Celex(legacy=True, verbose=True)
    # Do evaluation
    results = intrinsicEvaluation(tokenisers, do_whole_word=False, verbose=True)

    # Turn results into a file so that you can check them even if the terminal closes
    d = dict()
    for result in results:
        matrix = result.cm_morph
        pr, re, f1 = matrix.computePrReF1()
        tp, fp, tn, fn = matrix.compute()

        d[result.name] = {
            "morph-types": {
                "Pr": pr,
                "Re": re,
                "F1": f1,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn
            }
        }

    with open(TkTkTPaths.pathToEvaluations("morphology") / (dataset.identifier() + "-morphology_" + datetimeDashed() + ".json"), "w", encoding="utf-8") as handle:
        json.dump(d, handle)


##################################################################################################################


def constructTokenisers():
    return [
        Factory_KudoPiece(),
        Factory_BoMMaSum_BPE(),
        Factory_LeastToken_ULM(),
        Factory_LeastTokenThenProbability_ULM(),
    ]


def constructTokenisers_BPE():  # 43, 45.9, 53.2, 52.4
    return [
        Factory_BPE(),                    # Worst
        Factory_LeastToken_BPE(),         # Better by +1%
        Factory_BPEKnockout(),            # Best (+10% Pr, +25% Re)
        Factory_LeastToken_BPEKnockout()  # Second-best (+9% Pr, +17% Re). So, surprisingly, the gain from going from BPE to Viterbi-BPE is much smaller than the loss for going from BPE-knockout to Viterbi-BPE-knockout
    ]


def constructTokenisers_boundaryScores():
    a = -2
    b = +1
    generators = [BoundaryScoresChosen, BoundaryScoresAll]
    transforms = [LinearPT(a, b, False), LinearPT(a, b, True), PiecewisePT(a, b, False), PiecewisePT(a, b, True)]

    for g in generators:
        for t in transforms:
            yield Factory_BoMMaSum_FromTransform_ULM(g, t, VocabularyConstraintExact)


def constructTokenisers_boundaryScorePunishments():
    lower_bounds = [-0.25, -0.33, -0.5, -2, -3, -4]
    transforms = [LinearPT, PiecewisePT]
    generators = [BoundaryScoresChosen, BoundaryScoresAll]

    for low in lower_bounds:
        for t in transforms:
            for g in generators:
                yield Factory_BoMMaSum_FromTransform_ULM(g, t(low, +1), VocabularyConstraintExact)


def constructTokenisers_boundaryScoreLog():
    generators = [BoundaryScoresChosen, BoundaryScoresAll]
    for g in generators:
        yield Factory_BoMMaSum_FromTransform_ULM(g, LogPT(), VocabularyConstraintExact)


def constructTokenisers_hardBoundaryViterbi():
    generator_classes = [
        # BoundaryPrefixLength,
        # BoundaryPrefixLengthExtended,
        BoundarySuffixLength,
        BoundarySuffixLengthExtended,
        BoundaryPrefixAndSuffixLengthExtended
    ]
    constraint_classes = [
        VocabularyConstraintExact,
        # VocabularyConstraintAtLeastAll
    ]
    punishments = [0, -1, -2]
    normalisation = [False, True]

    for cc in constraint_classes:
        for gc in generator_classes:
            for p in punishments:
                for n in normalisation:
                    g = gc(punishment=p, do_normalise=n)
                    yield Factory_BoMMaSum_ULM(g, cc)


def constructTokenisers_suffixPunishments():
    punishments = [-0.5, -1, -1.5, -2, -2.5, -3, -3.5, -4, -4.5, -5, -5.5, -6, -6.5, -7, -7.5]
    for p in punishments:
        yield Factory_BoMMaSum_ULM(
            BoundarySuffixLength(punishment=p, do_normalise=True),
            VocabularyConstraintExact
        )


def constructTokenisers_leasttoken():
    return [
        Factory_LeastToken_ULM(),
        Factory_LeastTokenThenProbability_ULM(),
        Factory_ProbabilityThenLeastToken_ULM(),
    ]


def constructTokenisers_dropout():
    return [
        Factory_CanineBPEdropout(deterministic_threshold=None),
        Factory_CanineBPEdropout(deterministic_threshold=0.5),
        Factory_CanineBPEdropout(deterministic_threshold=0.4),
        Factory_CanineBPEdropout(deterministic_threshold=0.3),
        Factory_CanineBPEdropout(deterministic_threshold=0.2),
        Factory_CanineBPEdropout(deterministic_threshold=0.1)
    ]


def constructTokenisers_multiplicativeProbabilities():
    for scale in [0.25, 0.5, 0.75, 1.0, 1.1, 1.25]:
        yield Factory_BoMMaProduct(score_transform=PowerMBPT(power=1, scale=scale))
    yield Factory_BoMMaProduct(score_transform=DoublingMBPT())


def constructTokenisers_reBPE():
    for its in [1,2,3,4,5]:
        yield Factory_ReBPE(iterations=its)


if __name__ == "__main__":
    tokeniser_factories = constructTokenisers_suffixPunishments()
    ###
    evaluateTokenisers(f.buildTokeniser() for f in tokeniser_factories)
