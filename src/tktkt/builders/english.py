"""
Evaluate any tokeniser on English morphology.
"""
from transformers import CanineTokenizer, CanineForTokenClassification, AutoTokenizer
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


def getEnglishCANINE() -> CharacterClassifier:
    huggingface_checkpoint = PATH_CANINE_FOR_MBR_EN.as_posix()
    return HuggingFaceCharacterModelForTokenClassification(
        characters_to_modelinput=CanineTokenizer.from_pretrained(huggingface_checkpoint),
        for_token_classification=CanineForTokenClassification.from_pretrained(huggingface_checkpoint),
        input_kwargs={"padding": "max_length", "max_length": 4}
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
        files = getEnglishBpeFiles()
        only_for_vocabulary = BPEKnockout(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker
        )
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


class Builder_English_CanineViterbi_BPE(TokeniserBuilder[HFPointViterbi]):
    def buildTokeniser(self) -> T:
        english_bpe = getEnglishBpeFiles().toFastBPE()
        return HFPointViterbi(
            # HuggingFacePreprocessorForWords(robbert_tokenizer),  # The preprocessor that maps any string into the space of the vocabulary used.
            # vocab=robbert_tokenizer.get_vocab(),                 # The vocabulary that limits Viterbi steps.
            preprocessor=HuggingFacePreprocessorForWords(english_bpe),

            vocab=english_bpe.get_vocab(),
            max_step=20,
            vocabulary_constraint_class=VocabularyConstraintExact,
            score_generator_class=BoundaryScoresChosen,
            score_transform=LinearPT(-1, +1, negate_as_complement=True),

            huggingface_checkpoint=PATH_CANINE_FOR_MBR_EN.as_posix(),
            tokeniser_class=CanineTokenizer,
            model_class=CanineForTokenClassification,
            tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
        )


class Builder_English_CanineViterbi_ULM(TokeniserBuilder[HFPointViterbi]):
    """
    Build a Viterbi tokeniser with an underlying CANINE boundary probability model while choosing:
        - The grid generator that uses these probabilities;
        - The transformation applied to these probabilities;
        - The constraint put on steps afterwards, using the ULM vocabulary.
    """

    def __init__(self,
        generator: Type[ScoreGeneratorUsingCharacterClassifier]=BoundaryScoresChosen,
        score_transform: Optional[ProbabilityTransform]=LinearPT(-1, +1, negate_as_complement=False),
        constraint: Type[VocabularyConstraint]=VocabularyConstraintExact
    ):
        self.generator = generator
        self.score_transform = score_transform
        self.constraint = constraint

    def buildTokeniser(self) -> T:
        english_ulm = getEnglishKudo()

        return HFPointViterbi(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),

            vocab=english_ulm.get_vocab(),
            max_step=20,
            score_generator_class=self.generator,
            score_transform=self.score_transform,
            vocabulary_constraint_class=self.constraint,

            huggingface_checkpoint=PATH_CANINE_FOR_MBR_EN.as_posix(),
            tokeniser_class=CanineTokenizer,
            model_class=CanineForTokenClassification,
            tokeniser_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
        )


class Builder_English_LeastTokenThenHF_ULM(TokeniserBuilder[LeastTokenViterbiWithProbabilityTiebreaker]):
    def buildTokeniser(self) -> T:
        kudo: HuggingFaceTokeniser = Builder_English_KudoPiece().buildTokeniser()
        vocab = kudo.backend.get_vocab()
        assert isinstance(vocab, dict)

        classifier = getEnglishCANINE()
        return LeastTokenViterbiWithProbabilityTiebreaker(
            preprocessor=kudo.preprocessor,
            vocab=vocab,
            max_step=20,
            logprob_classifier=classifier
        )


class Builder_English_HfThenLeastToken_ULM(TokeniserBuilder[ProbabilityViterbiWithLeastTokenTiebreaker]):
    def buildTokeniser(self) -> T:
        kudo: HuggingFaceTokeniser = Builder_English_KudoPiece().buildTokeniser()
        vocab = kudo.backend.get_vocab()
        assert isinstance(vocab, dict)

        classifier = getEnglishCANINE()
        return ProbabilityViterbiWithLeastTokenTiebreaker(
            preprocessor=kudo.preprocessor,
            vocab=vocab,
            max_step=20,
            logprob_classifier=classifier
        )


class Builder_English_CanineBPEdropout(TokeniserBuilder[GuidedBPEDropout]):

    def __init__(self, deterministic_threshold: float=None):
        self.threshold = deterministic_threshold

    def buildTokeniser(self) -> T:
        english_bpe_files = getEnglishBpeFiles()
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


class Builder_English_CanineViterbiMultiplicative_ULM(TokeniserBuilder[MultiplicativeBalanceViterbi]):
    def __init__(self, score_transform: MultiplicativelyBalancedProbabilityTransform=DoublingMBPT()):
        self.transform = score_transform

    def buildTokeniser(self):
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()

        return MultiplicativeBalanceViterbi(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            vocab=english_ulm.get_vocab(),
            max_step=20,

            logprob_classifier=english_canine_mbr,
            transform=self.transform
        )
