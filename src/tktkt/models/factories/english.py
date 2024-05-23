"""
Evaluate any tokeniser on English morphology.

TODO: Are you sure all these models have proper preprocessing, with byte-level stuff and G for BPE and _ for ULM?
"""
from typing import Type, Optional

from transformers import CanineTokenizer, CanineForTokenClassification, AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from bpe_knockout.project.config import KnockoutDataConfiguration, setupEnglish, Pℛ𝒪𝒥ℰ𝒞𝒯
from bpe_knockout.auxiliary.tokenizer_interface import BpeTokeniserPath

from ...preparation.instances import HuggingFacePreprocessorForWords, RobertaSpaceMarker, CommonsensePreprocessor
from ...models.viterbi.instances import *
from ...models.viterbi.objectives_guided import *
from ...models.viterbi.objectives_postprocessors import *
from ...models.huggingface.wrapper import HuggingFaceTokeniser
from ...models.bpe.knockout import BPEKnockout
from ...files.paths import relativeToCwd, DataPaths
from ...interfaces.tokeniser import Tokeniser

from .base import TokeniserFactory


PATH_CANINE_FOR_MBR_EN = relativeToCwd(DataPaths.pathToCheckpoints() / "CANINE-C_MBR-en_2024-02-12_19-35-28")


def getEnglishBpeFiles() -> BpeTokeniserPath:
    """
    Accessing BPE this way ensures that when you do knockout or you strip the HuggingFace tokeniser's pretokeniser,
    other constructors are unaffected.
    """
    with KnockoutDataConfiguration(setupEnglish()):
        return Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser


def getEnglishKudo() -> AlbertTokenizerFast:
    return AutoTokenizer.from_pretrained("albert/albert-base-v2")


def getEnglishCANINE() -> CharacterClassifier:
    huggingface_checkpoint = PATH_CANINE_FOR_MBR_EN.as_posix()
    return HuggingFaceCharacterModelForTokenClassification(
        characters_to_modelinput=CanineTokenizer.from_pretrained(huggingface_checkpoint),
        for_token_classification=CanineForTokenClassification.from_pretrained(huggingface_checkpoint),
        input_kwargs={"padding": "max_length", "max_length": 4}
    )


class Factory_English_BPE(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        english_bpe = getEnglishBpeFiles().toFastBPE()  # HuggingFace automatically sets a ByteBased tokenizers.pretokeniser on all RobertaTokenizerFast instances, which also implicitly adds a start-of-word Ġ as replacement for spaces.
        return HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)


class Factory_English_BPEKnockout(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        files = getEnglishBpeFiles()
        return BPEKnockout(
            preprocessor=CommonsensePreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker
        )


class Factory_English_KudoPiece(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        tk = getEnglishKudo()
        return HuggingFaceTokeniser(tk, for_single_words=True)


class Factory_English_CompressiveViterbi_BPE(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        english_bpe = getEnglishBpeFiles().toFastBPE()
        return LeastTokenViterbi(
            preprocessor=HuggingFacePreprocessorForWords(english_bpe),
            vocab=english_bpe.get_vocab(),
            max_step=20
        )


class Factory_English_CompressiveViterbi_BPEKnockout(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
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


class Factory_English_CompressiveViterbi_ULM(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        hf_english_ulm = getEnglishKudo()
        return LeastTokenViterbi(
            HuggingFacePreprocessorForWords(hf_english_ulm),
            vocab=hf_english_ulm.get_vocab(),
            max_step=20
        )


class Factory_English_CanineViterbi_BPE(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
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


class Factory_English_CanineViterbi_ULM(TokeniserFactory):

    def __init__(self,
        generator: Type[ScoreGeneratorUsingCharacterClassifier]=BoundaryScoresChosen,
        score_transform: Optional[ProbabilityTransform]=LinearPT(-1, +1, negate_as_complement=False),
        constraint: Type[VocabularyConstraint]=VocabularyConstraintExact
    ):
        self.generator = generator
        self.score_transform = score_transform
        self.constraint = constraint

    def buildTokeniser(self) -> Tokeniser:
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


class Factory_English_LeastTokenThenHF_ULM(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        kudo: HuggingFaceTokeniser = Factory_English_KudoPiece().buildTokeniser()
        classifier = getEnglishCANINE()
        return LeastTokenViterbiWithProbabilityTiebreaker(
            preprocessor=kudo.preprocessor,
            vocab=kudo.getVocabMapping(),
            max_step=20,
            logprob_classifier=classifier
        )


class Factory_English_HfThenLeastToken_ULM(TokeniserFactory):
    def buildTokeniser(self) -> Tokeniser:
        kudo: HuggingFaceTokeniser = Factory_English_KudoPiece().buildTokeniser()
        classifier = getEnglishCANINE()
        return ProbabilityViterbiWithLeastTokenTiebreaker(
            preprocessor=kudo.preprocessor,
            vocab=kudo.getVocabMapping(),
            max_step=20,
            logprob_classifier=classifier
        )
