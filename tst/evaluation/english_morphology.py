"""
Evaluate any tokeniser on English morphology.

TODO: To be tested:
    - English BPE-knockout-reify
    - We have prefix objectives. What about suffix objectives? I.e.: if you have splits |abcde|fghi| then you get a
      score of 3 for a token [CDE] because it ENDS on a boundary, not starts on it.
      And could you do prefix and suffix combined? Giving score for starting and ending on a boundary, proportional to
      the distance travelled to/from that point and capped by the distance to the nearest boundary in that direction?

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

from bpe_knockout.project.config import KnockoutDataConfiguration, setupEnglish

from tktkt.builders.english import *
from tktkt.util.timing import datetimeDashed
from tktkt.evaluation.morphological import intrinsicEvaluation
from tktkt.models.viterbi.objectives_guided import *
from tktkt.models.viterbi.objectives_postprocessors import *
from tktkt.files.paths import DataPaths


def testTokenisers(tokenisers: Iterable[Tokeniser]):
    with KnockoutDataConfiguration(setupEnglish()):
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

        with open(DataPaths.pathToEvaluations() / (Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag() + "_morphology_" + datetimeDashed() + ".json"), "w", encoding="utf-8") as handle:
            json.dump(d, handle)

##################################################################################################################

def constructTokenisers():
    return [
        Builder_English_KudoPiece(),
        Builder_English_CanineViterbi_BPE(),
        Builder_English_LeastToken_ULM(),
        Builder_English_LeastTokenThenHF_ULM(),
    ]


def constructTokenisers_BPE():  # 43, 45.9, 53.2, 52.4
    return [
        Builder_English_BPE(),                            # Worst
        Builder_English_LeastToken_BPE(),         # Better by +1%
        Builder_English_BPEKnockout(),                    # Best (+10% Pr, +25% Re)
        Builder_English_LeastToken_BPEKnockout()  # Second-best (+9% Pr, +17% Re). So, surprisingly, the gain from going from BPE to Viterbi-BPE is much smaller than the loss for going from BPE-knockout to Viterbi-BPE-knockout
    ]


def constructTokenisers_prefixGenerators():
    generators = [
        HardBoundaryPrefixLength,
        HardBoundaryPrefixLengthExtended,
        HardBoundaryAndNonBoundaryPrefixLength,
        HardBoundaryAndNonBoundaryPrefixLengthExtended
    ]
    constraints = [
        VocabularyConstraintExact,
        VocabularyConstraintAtLeastAll
    ]

    for c in constraints:
        for g in generators:
            yield Builder_English_CanineViterbi_ULM(g, None, c)


def constructTokenisers_boundaryScores():
    a = -2
    b = +1
    generators = [BoundaryScoresChosen, BoundaryScoresAll]
    transforms = [LinearPT(a, b, False), LinearPT(a, b, True), PiecewisePT(a, b, False), PiecewisePT(a, b, True)]

    for g in generators:
        for t in transforms:
            yield Builder_English_CanineViterbi_ULM(g, t, VocabularyConstraintExact)


def constructTokenisers_boundaryScorePunishments():
    lower_bounds = [-0.25, -0.33, -0.5, -2, -3, -4]
    transforms = [LinearPT, PiecewisePT]
    generators = [BoundaryScoresChosen, BoundaryScoresAll]

    for low in lower_bounds:
        for t in transforms:
            for g in generators:
                yield Builder_English_CanineViterbi_ULM(g, t(low, +1), VocabularyConstraintExact)


def constructTokenisers_boundaryScoreLog():
    generators = [BoundaryScoresChosen, BoundaryScoresAll]
    for g in generators:
        yield Builder_English_CanineViterbi_ULM(g, LogPT(), VocabularyConstraintExact)


def constructTokenisers_leasttoken():
    return [
        Builder_English_LeastToken_ULM(),
        Builder_English_LeastTokenThenHF_ULM(),
        Builder_English_HfThenLeastToken_ULM(),
    ]


def constructTokenisers_dropout():
    return [
        Builder_English_CanineBPEdropout(None),
        Builder_English_CanineBPEdropout(0.5),
        Builder_English_CanineBPEdropout(0.4),
        Builder_English_CanineBPEdropout(0.3),
        Builder_English_CanineBPEdropout(0.2),
        Builder_English_CanineBPEdropout(0.1)
    ]


if __name__ == "__main__":
    tokeniser_builders = constructTokenisers_dropout()
    ###
    testTokenisers(builder.buildTokeniser() for builder in tokeniser_builders)
