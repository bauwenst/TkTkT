"""
Evaluate any tokeniser on English morphology.

TODO: To be tested:
    - English BPE-knockout-reify
    - You have run the 4 joint versions of (-2,+1) "symmetric" probability on VSC.
      Now implement and run the 4 boundary-only versions, and see which ones are equivalent and which are not.
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
import itertools
import json

from tktkt.util.timing import datetimeDashed, timeit
from typing import Type, Optional

from transformers import CanineTokenizer, CanineForTokenClassification, AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from bpe_knockout.project.config import TemporaryContext, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯

from tktkt.preparation.instances import HuggingFacePreprocessorForWords
from tktkt.evaluation.morphological import intrinsicEvaluation
from tktkt.models.viterbi.instances import HFPointViterbi, LeastTokenViterbi, LeastTokenWithHfTiebreaker
from tktkt.models.viterbi.objectives_guided import *
from tktkt.models.viterbi.objectives_postprocessors import *
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser
from tktkt.files.paths import relativeToCwd, DataPaths

from tst.preamble import *


with TemporaryContext(setupEnglish()):
    english_bpe = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser.toFastBPE()  # Has a byte-based preprocessor; HuggingFace sets it automatically on all Roberta tokenisers.


def make_EnglishBPE():
    return HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)


def make_EnglishKudoPiece():
    tk: AlbertTokenizerFast = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    return HuggingFaceTokeniser(tk, for_single_words=True)


def make_CompressiveViterbiBPE():
    return LeastTokenViterbi(
        HuggingFacePreprocessorForWords(english_bpe),
        vocab=english_bpe.get_vocab(),
        max_step=20
    )


def make_CompressiveViterbiULM():
    english_ulm: AlbertTokenizerFast = AutoTokenizer.from_pretrained("albert/albert-base-v2")

    return LeastTokenViterbi(
        HuggingFacePreprocessorForWords(english_ulm),
        vocab=english_ulm.get_vocab(),
        max_step=20
    )


def make_CanineViterbiBPE():
    return HFPointViterbi(
        # HuggingFacePreprocessorForWords(robbert_tokenizer),  # The preprocessor that maps any string into the space of the vocabulary used.
        # vocab=robbert_tokenizer.get_vocab(),                 # The vocabulary that limits Viterbi steps.
        preprocessor=HuggingFacePreprocessorForWords(english_bpe),

        vocab=english_bpe.get_vocab(),
        max_step=20,
        vocabulary_constraint_class=VocabularyConstraintExact,
        score_generator_class=BoundaryScoresChosen,
        score_transform=LinearPT(-1, +1, negate_as_complement=True),

        huggingface_checkpoint=relativeToCwd(DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28").as_posix(),
        tokeniser_class=CanineTokenizer,
        model_class=CanineForTokenClassification,
        tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
    )


def make_CanineViterbiULM(
        generator: Type[ScoreGeneratorUsingCharacterClassifier]=BoundaryScoresChosen,
        score_transform: Optional[ProbabilityTransform]=LinearPT(-1, +1, negate_as_complement=False),
        constraint: Type[VocabularyConstraint]=VocabularyConstraintExact
    ):
    english_ulm: AlbertTokenizerFast = AutoTokenizer.from_pretrained("albert/albert-base-v2")

    return HFPointViterbi(
        preprocessor=HuggingFacePreprocessorForWords(english_ulm),

        vocab=english_ulm.get_vocab(),
        max_step=20,
        score_generator_class=generator,
        score_transform=score_transform,
        vocabulary_constraint_class=constraint,

        huggingface_checkpoint=relativeToCwd(DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28").as_posix(),
        tokeniser_class=CanineTokenizer,
        model_class=CanineForTokenClassification,
        tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
    )


def make_LeastTokenTiebroken():
    canine_viterbi = make_CanineViterbiULM()
    constraint: VocabularyConstraint = canine_viterbi.objectives[0].score_generator

    vocab = constraint.vocab
    classifier = constraint.nested_generator.logprob_classifier
    return LeastTokenWithHfTiebreaker(
        preprocessor=canine_viterbi.preprocessor,
        vocab=vocab,
        max_step=20,
        logprob_classifier=classifier
    )


@timeit
def constructTokenisers():
    return [
        # make_EnglishBPE(),
        # make_EnglishKudoPiece()
        # make_CompressiveViterbiULM(),
        # make_CanineViterbiBPE(),
        make_LeastTokenTiebroken()
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
            yield make_CanineViterbiULM(g, None, c)


def constructTokenisers_boundaryScores():
    a = -2
    b = +1
    generators = [BoundaryScoresChosen, BoundaryScoresAll]
    transforms = [LinearPT(a, b, False), LinearPT(a, b, True), PiecewisePT(a, b, False), PiecewisePT(a, b, True)]

    for g in generators:
        for t in transforms:
            yield make_CanineViterbiULM(g, t, VocabularyConstraintExact)


def constructTokenisers_boundaryScorePunishments():
    lower_bounds = [-0.25, -0.33, -0.5, -2, -3, -4]
    transforms = [LinearPT, PiecewisePT]
    generators = [BoundaryScoresChosen, BoundaryScoresAll]

    for low in lower_bounds:
        for t in transforms:
            for g in generators:
                yield make_CanineViterbiULM(g, t(low, +1), VocabularyConstraintExact)


def constructTokenisers_boundaryScoreLog():
    generators = [BoundaryScoresChosen, BoundaryScoresAll]
    for g in generators:
        yield make_CanineViterbiULM(g, LogPT(), VocabularyConstraintExact)


if __name__ == "__main__":
    with TemporaryContext(setupEnglish()):
        # Do evaluation
        results = intrinsicEvaluation(constructTokenisers(), do_whole_word=False, verbose=True)

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

        with open(DataPaths.pathToEvaluations() / (Pℛ𝒪𝒥ℰ𝒞𝒯.config.langTag() + "morphology_" + datetimeDashed() + ".json"), "w", encoding="utf-8") as handle:
            json.dump(d, handle)
