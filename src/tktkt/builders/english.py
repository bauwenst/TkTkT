"""
Evaluate any tokeniser on English morphology.
"""
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from bpe_knockout.project.config import KnockoutDataConfiguration, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯, defaultTokeniserFiles
from bpe_knockout.auxiliary.tokenizer_interface import BpeTokeniserPath

from ..preparation.instances import *
from ..models.viterbi.instances import *
from ..models.huggingface.wrapper import HuggingFaceTokeniser
from ..models.bpe.base import ClassicBPE
from ..models.bpe.knockout import BPEKnockout, ReBPE
from ..models.bpe.guided import GuidedBPEDropout
from ..models.ngram.alphabet import UnicodeTokeniser
from ..files.paths import relativeToCwd, TkTkTPaths
from .base import TokeniserBuilder, T


PATH_CANINE_FOR_MBR_EN = relativeToCwd(TkTkTPaths.pathToCheckpoints() / "CANINE-C_MBR-en_2024-02-12_19-35-28")


def getEnglishBpeFiles() -> BpeTokeniserPath:
    """
    Accessing BPE this way ensures that when you do knockout or you strip the HuggingFace tokeniser's pretokeniser,
    other constructors are unaffected.
    """
    with KnockoutDataConfiguration(setupEnglish()):
        return defaultTokeniserFiles()


def getEnglishKudo() -> AlbertTokenizerFast:
    return AutoTokenizer.from_pretrained("albert/albert-base-v2")


def getEnglishCANINE() -> HuggingFaceForBinaryCharacterClassification:
    return HuggingFaceForBinaryCharacterClassification(
        characterclassifier_checkpoint=PATH_CANINE_FOR_MBR_EN.as_posix(),
        input_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
    )


class Builder_English_BPE(TokeniserBuilder[HuggingFaceTokeniser]):
    def buildTokeniser(self) -> T:
        english_bpe = getEnglishBpeFiles().toFastBPE()  # HuggingFace automatically sets a ByteBased tokenizers.pretokeniser on all RobertaTokenizerFast instances, which also implicitly adds a start-of-word Ġ as replacement for spaces.
        return HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)


class Builder_English_BPE_native(TokeniserBuilder[ClassicBPE]):
    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return ClassicBPE(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            boundary_marker=RobertaSpaceMarker
        )


class Builder_English_BPEKnockout(TokeniserBuilder[BPEKnockout]):
    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return BPEKnockout(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker
        )


class Builder_English_ReBPE(TokeniserBuilder[ReBPE]):
    def __init__(self, iterations: int, reduced: bool=False):
        self.its = iterations
        self.bc = reduced

    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return ReBPE(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker,

            iterations=self.its,
            backwards_compatible=self.bc
        )


class Builder_English_KudoPiece(TokeniserBuilder[HuggingFaceTokeniser]):
    def buildTokeniser(self) -> T:
        tk = getEnglishKudo()
        return HuggingFaceTokeniser(tk, for_single_words=True)


class Builder_English_LeastToken_BPE(TokeniserBuilder[LeastTokenViterbi]):
    def buildTokeniser(self) -> T:
        english_bpe = getEnglishBpeFiles().toFastBPE()
        return LeastTokenViterbi(
            preprocessor=HuggingFacePreprocessorForWords(english_bpe),
            vocab=english_bpe.get_vocab(),
            max_step=20
        )


class Builder_English_LeastToken_BPEKnockout(TokeniserBuilder[LeastTokenViterbi]):
    def buildTokeniser(self) -> T:
        # Get starting BPE vocabulary
        files = getEnglishBpeFiles()

        # Prune the vocabulary with BPE-knockout
        only_for_vocabulary = BPEKnockout(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker
        )

        # Use this new vocabulary
        return LeastTokenViterbi(
            preprocessor=only_for_vocabulary.preprocessor,
            vocab=only_for_vocabulary.vocab,
            max_step=20
        )


class Builder_English_LeastToken_ULM(TokeniserBuilder[LeastTokenViterbi]):
    def buildTokeniser(self) -> T:
        hf_english_ulm = getEnglishKudo()
        return LeastTokenViterbi(
            HuggingFacePreprocessorForWords(hf_english_ulm),
            vocab=hf_english_ulm.get_vocab(),
            max_step=20
        )


class Builder_English_BoMMaSum_BPE(TokeniserBuilder[BoMMa_Sum]):
    def buildTokeniser(self) -> T:
        english_bpe        = getEnglishBpeFiles().toFastBPE()
        english_canine_mbr = getEnglishCANINE()
        return BoMMa_Sum(
            preprocessor=HuggingFacePreprocessorForWords(english_bpe),
            max_step=20,

            score_generator=BoundaryScoresChosen(LinearPT(-1, +1, negate_as_complement=True)),

            vocab=english_bpe.get_vocab(),
            vocabulary_constraint_class=VocabularyConstraintExact,
        ).from_object(english_canine_mbr)


class Builder_English_BoMMaSum_ULM(TokeniserBuilder[BoMMa_Sum]):
    """
    Build a Viterbi tokeniser with an underlying CANINE boundary probability model while choosing:
        - The grid generator that uses these probabilities;
        - The transformation applied to these probabilities;
        - The constraint put on steps afterwards, using the ULM vocabulary.
    """

    def __init__(self, generator: ScoreGeneratorUsingCharacterClassifier, constraint: Type[VocabularyConstraint]):
        self.generator = generator
        self.constraint = constraint

    def buildTokeniser(self) -> T:
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()
        return BoMMa_Sum(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            max_step=20,

            score_generator=self.generator,

            vocabulary_constraint_class=self.constraint,
            vocab=english_ulm.get_vocab()
        ).from_object(english_canine_mbr)


class Builder_English_BoMMaSum_FromTransform_ULM(Builder_English_BoMMaSum_ULM):
    def __init__(self,
        generator: Type[ScoreGeneratorUsingCharacterClassifierForTransform]=BoundaryScoresChosen,
        score_transform: ProbabilityTransform=LinearPT(-1, +1, negate_as_complement=False),
        constraint: Type[VocabularyConstraint]=VocabularyConstraintExact
    ):
        super().__init__(generator(score_transform), constraint)


class Builder_English_BoMMaProduct_ULM(TokeniserBuilder[BoMMa_Product]):
    """
    Instantiates a BoMMa_Product tokeniser specifically with a BoundaryScoresChosen generator and a multiplicatively
    balanced probability transform (a small subset of all BoMMa_Product tokenisers, technically).

    The idea is this: if you have a boundary model that gives you probabilities at each character boundary, one thing
    you might do to score a segmentation is multiply the probabilities of the boundaries you choose to split on.
    The problem here is that it disincentivises gathering a higher amount of high-probability boundaries even if it takes
    no other bad splits to do so: 0.9999 * 0.9999 * 0.9999 is smaller than 0.9999 * 0.9999. A second issue is that when
    you do hit a low-probability boundary, you carry that forever: 0.1 * 0.9999 * 0.9999 * 0.9999 * 0.9999 * 0.9999
    has 5 perfect splits and 1 bad split, and it is 10x smaller than having even 1 good split at 0.9999.

    So, what you want is to give all probabilities above 50% a boost so that you can recover completely from a bad split
    and so that two perfect splits are better than one perfect split.
    """

    def __init__(self, score_transform: MultiplicativelyBalancedProbabilityTransform=DoublingMBPT()):
        self.balanced_transform = score_transform

    def buildTokeniser(self):
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()
        return BoMMa_Product(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            max_step=20,

            score_generator=BoundaryScoresChosen(self.balanced_transform),

            vocabulary_constraint_class=VocabularyConstraintExact,
            vocab=english_ulm.get_vocab()
        ).from_object(english_canine_mbr)


class Builder_English_LeastTokenThenProbability_ULM(TokeniserBuilder[LeastTokenViterbiWithProbabilityTiebreaker]):
    def buildTokeniser(self) -> T:
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()
        return LeastTokenViterbiWithProbabilityTiebreaker(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            max_step=20,

            vocab=english_ulm.get_vocab(),
            logprob_classifier=english_canine_mbr
        )


class Builder_English_ProbabilityThenLeastToken_ULM(TokeniserBuilder[ProbabilityViterbiWithLeastTokenTiebreaker]):
    def buildTokeniser(self) -> T:
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()
        return ProbabilityViterbiWithLeastTokenTiebreaker(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            max_step=20,

            vocab=english_ulm.get_vocab(),
            logprob_classifier=english_canine_mbr
        )


class Builder_English_CanineBPEdropout(TokeniserBuilder[GuidedBPEDropout]):

    def __init__(self, deterministic_threshold: float=None):
        self.threshold = deterministic_threshold

    def buildTokeniser(self) -> T:
        english_bpe_files  = getEnglishBpeFiles()
        english_canine_mbr = getEnglishCANINE()
        return GuidedBPEDropout(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),  # We know the BPE files uses this marker, so we can manually specify it.

            vocab=english_bpe_files.loadVocabulary(),
            merges=english_bpe_files.loadMerges(),
            boundary_marker=RobertaSpaceMarker,

            dropout_probability=english_canine_mbr,
            always_dropout_above=self.threshold,
        )


class Builder_English_Character(TokeniserBuilder[UnicodeTokeniser]):
    def buildTokeniser(self) -> T:
        return UnicodeTokeniser(preprocessor=IdentityPreprocessor)
