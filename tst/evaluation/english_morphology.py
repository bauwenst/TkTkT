"""
Evaluate any tokeniser on English morphology.

TODO: To be tested:
    - English BPE-knockout-reify
    - Re-test all my prefix objectives: with punishment, extended, with AtLeastAll constraint...
        - That's going to be a good 2*2*2 = 8 tokenisers. Better set up a VSC loop for that.

    - CANINE+Viterbi variations:
        x Symmetric probability objective (the best for ULM vocab) not with exact constraint, but AtLeastAll.
        x Hard prefix objective with AtLeastAll constraint.
        - Hard prefix objective with punishment for bad splits.
            - What we HAVE is an incentive to not use short tokens when starting on a boundary.
            - What we DON'T HAVE is an incentive to not use many tokens when not starting on a boundary. You have to bridge the gap to the next boundary ASAP!
            - Also, another explanation for why the prefix score performs worse than boundary matching is that you give NO REWARD when Viterbi wants to use a step that starts on a boundary but jumps over the next boundary, which would still count as >0 score when boundary matching.

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
from tktkt.util.timing import timeit
from typing import Type

from transformers import CanineTokenizer, CanineForTokenClassification, AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from bpe_knockout.project.config import TemporaryContext, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯

from tktkt.preparation.instances import HuggingFacePreprocessorForWords
from tktkt.evaluation.morphological import intrinsicEvaluation
from tktkt.models.viterbi.instances import HFPointViterbi, LeastTokenViterbi
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
        score_generator_class=BoundaryAndNonBoundaryLogProbability,

        huggingface_checkpoint=relativeToCwd(DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28").as_posix(),
        tokeniser_class=CanineTokenizer,
        model_class=CanineForTokenClassification,
        tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
    )


def make_CanineViterbiULM(generator: Type[ScoreGeneratorUsingCharacterClassifier]=SymmetricBoundaryProbability, constraint: Type[VocabularyConstraint]=VocabularyConstraintExact):
    english_ulm: AlbertTokenizerFast = AutoTokenizer.from_pretrained("albert/albert-base-v2")

    return HFPointViterbi(
        preprocessor=HuggingFacePreprocessorForWords(english_ulm),

        vocab=english_ulm.get_vocab(),
        max_step=20,
        score_generator_class=generator,
        vocabulary_constraint_class=constraint,

        huggingface_checkpoint=relativeToCwd(DataPaths.pathToCheckpoints() / "CANINE-C_2024-02-12_19-35-28").as_posix(),
        tokeniser_class=CanineTokenizer,
        model_class=CanineForTokenClassification,
        tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
    )


@timeit
def constructTokenisers():
    return [
        # make_EnglishBPE(),
        # make_EnglishKudoPiece()
        # make_CompressiveViterbiULM(),
        # make_CanineViterbiBPE(),
        make_CanineViterbiULM(g, v) for g,v in itertools.product([
            HardBoundaryPrefixLength,
            HardBoundaryPrefixLengthExtended,
            HardBoundaryAndNonBoundaryPrefixLength,
            HardBoundaryAndNonBoundaryPrefixLengthExtended
        ], [
            VocabularyConstraintExact,
            VocabularyConstraintAtLeastAll
        ])
    ]


def constructTokenisers2():
    viterbi_tokeniser = make_CanineViterbiULM(SymmetricBoundaryAndNonBoundaryProbability, VocabularyConstraintExact)

    a = -2
    b = +1
    transforms = [LinearPT_TrueComp(a, b), LinearPT_NegComp(a, b), PiecewisePT_TrueComp(a, b), PiecewisePT_NegComp(a, b)]
    for t in transforms:
        print("SWITCHING TO TRANSFORM:", t)
        viterbi_tokeniser.objectives[0].score_generator.nested_generator.T = t
        yield viterbi_tokeniser


if __name__ == "__main__":
    with TemporaryContext(setupEnglish()):
        intrinsicEvaluation(constructTokenisers2(), do_whole_word=True, verbose=True)
