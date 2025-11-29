"""
Evaluate any tokeniser on English morphology.
"""
from ..interfaces import TokeniserFactory, Deserialiser
from ..interfaces.identifiers import WithSpecials
from ..models.predictive.viterbi.instances import *
from ..models.bpe.base import ClassicBPE
from ..models.bpe.knockout import BPEKnockout, ReBPE
from ..models.bpe.guided import GuidedBPEDropout
from ..models.huggingface.bpe import HuggingFaceBPETokeniser
from ..models.kudopiece.segmentation import KudoPieceTokeniser
from ..models.random.grampa import GRaMPa, PowerNormalisation
from ..models.ngram.alphabet import UnicodeTokeniser
from ..wrappers.multiplexing import StochasticTokeniserSwitch, MultiplexedPreprocessor
from .preprocessing import *
from .deserialisation import *
from .deserialisation import BPE_Deserialiser, KudoPiece_Deserialiser, detectBoundaryMarkerFromVocabulary, getEnglishCANINE


########################################################################################################################


class Factory_BPE(TokeniserFactory[HuggingFaceBPETokeniser[WithSpecials]]):
    def __init__(self, preprocessor: Preprocessor=None, dropout: float=0.0, files: BPE_Deserialiser=BPE32ki_SlimPajama3M()):
        self._prep = preprocessor
        self._dropout = dropout
        self._files = files

    def buildTokeniser(self):
        if self._prep is None:
            self._prep = self._files.preprocessorEffective()  # Effective preprocessor because we use HuggingFace inference, which does not do built-in preprocessing.

        return HuggingFaceBPETokeniser(vocab=self._files.buildVocabulary(), merges=self._files.buildMerges(), dropout=self._dropout, preprocessor=self._prep)
        # english_bpe = getEnglishBpeFiles().toFastBPE()  # HuggingFace automatically sets a ByteBased tokenizers.pretokeniser on all RobertaTokenizerFast instances, which also implicitly adds a start-of-word Ġ as replacement for spaces.
        # return HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)


class Factory_BPE_Pythonic(TokeniserFactory[ClassicBPE[WithSpecials]]):
    def __init__(self, preprocessor: Preprocessor=None, files: BPE_Deserialiser[WithSpecials]=BPE32ki_SlimPajama3M()):
        self._prep = preprocessor
        self._files = files

    def buildTokeniser(self):
        if self._prep is None:
            self._prep = self._files.preprocessorEffective()

        return ClassicBPE(
            preprocessor=self._prep,
            vocab=self._files.buildVocabulary(),
            merges=self._files.buildMerges()
        )


class Factory_BPEKnockout(TokeniserFactory[BPEKnockout]):
    def __init__(self, preprocessor: Preprocessor=None, files: BPE_Deserialiser=BPE32ki_SlimPajama3M()):
        self._prep = preprocessor
        self._files = files

    def buildTokeniser(self):
        return BPEKnockout(
            preprocessor=self._prep,
            vocab=self._files.buildVocabulary(),
            merges=self._files.buildMerges(),
            language="English"
        )


class Factory_ReBPE(TokeniserFactory[ReBPE]):
    def __init__(self, iterations: int, reduced: bool=False, preprocessor: Preprocessor=None, files: BPE_Deserialiser=BPE32ki_SlimPajama3M()):
        self._prep = preprocessor
        self._files = files
        self.its = iterations
        self.bc = reduced

    def buildTokeniser(self):
        if self._prep is None:
            self._prep = self._files.preprocessorEffective()

        return ReBPE(
            preprocessor=self._prep,
            vocab=self._files.buildVocabulary(),
            merges=self._files.buildMerges(),
            language="English",

            iterations=self.its,
            backwards_compatible=self.bc
        )


class Factory_KudoPiece(TokeniserFactory[KudoPieceTokeniser[WithSpecials]]):
    """
    Defaults to the 32k SlimPajama vocab.
    """

    def __init__(self, preprocessor: Preprocessor=None, files: KudoPiece_Deserialiser=KudoPiece32ki_SlimPajama3M(), kbest: int=64, alpha: float=1.0):
        self._prep = preprocessor
        self._files = files
        self._kbest = kbest
        self._alpha = alpha

    def buildTokeniser(self):
        if self._prep is None:
            self._prep = self._files.preprocessorNative()  # Native preprocessor due to SentencePiece inference.

        return KudoPieceTokeniser(
            preprocessor=self._prep,
            model_file=self._files.getModelFile(),
            vocab=self._files.buildVocabulary(),

            kbest=self._kbest,
            smoothing_power=self._alpha
        )


class Factory_LeastToken(TokeniserFactory[LeastTokenViterbi]):
    def __init__(self, preprocessor: Preprocessor=None, files: Deserialiser=BPE32ki_SlimPajama3M()):
        self._files = files
        self._prep = preprocessor or files.preprocessorEffective()

    def buildTokeniser(self):
        return LeastTokenViterbi(
            preprocessor=self._prep,
            vocab=self._files.buildVocabulary(),
            max_step=20
        )


class Factory_LeastToken_BPEKnockout(TokeniserFactory[LeastTokenViterbi]):
    def __init__(self, preprocessor: Preprocessor=None, files: BPE_Deserialiser=BPE32ki_SlimPajama3M()):
        self._files = files
        self._prep = preprocessor or files.preprocessorEffective()

    def buildTokeniser(self):
        # Prune the vocabulary with BPE-knockout
        only_for_vocabulary = BPEKnockout(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),
            vocab=self._files.buildVocabulary(),
            merges=self._files.buildMerges(),
            language="English"
        )

        # Use this new vocabulary
        return LeastTokenViterbi(
            preprocessor=only_for_vocabulary.preprocessor,
            vocab=only_for_vocabulary.vocab,
            max_step=20
        )


class Factory_BoMMaSum(TokeniserFactory[BoMMa_Sum]):
    """
    Build a Viterbi tokeniser with an underlying CANINE boundary probability model while choosing:
        - The grid generator that uses these probabilities;
        - The transformation applied to these probabilities;
        - The constraint put on steps afterwards, using the ULM vocabulary.
    """

    def __init__(self,
                 generator: ScoreGeneratorUsingCharacterClassifier=BoundaryScoresChosen(LinearPT(-1, +1, negate_as_complement=True)),
                 constraint: Type[VocabularyConstraint]=VocabularyConstraintExact,
                 preprocessor: Preprocessor=None, files: Deserialiser=KudoPiece32ki_SlimPajama3M()):
        self._files = files
        self._prep = preprocessor  # TODO: As we know, BoMMa has a two-preprocessor problem.

        self.generator = generator
        self.constraint = constraint

    def buildTokeniser(self):
        english_canine_mbr = getEnglishCANINE()
        return BoMMa_Sum(
            preprocessor=self._prep,
            max_step=20,

            score_generator=self.generator,

            vocabulary_constraint_class=self.constraint,
            vocab=self._files.buildVocabulary()
        ).from_object(english_canine_mbr)


class Factory_BoMMaSum_FromTransform(Factory_BoMMaSum):
    def __init__(self,
        files: Deserialiser=KudoPiece32ki_SlimPajama3M(),
        generator: Type[ScoreGeneratorUsingCharacterClassifierForTransform]=BoundaryScoresChosen,
        score_transform: ProbabilityTransform=LinearPT(-1, +1, negate_as_complement=False),
        constraint: Type[VocabularyConstraint]=VocabularyConstraintExact
    ):
        super().__init__(generator=generator(score_transform), constraint=constraint, files=files)


class Factory_BoMMaProduct(TokeniserFactory[BoMMa_Product]):
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

    def __init__(self, files: Deserialiser=KudoPiece32ki_SlimPajama3M(), score_transform: MultiplicativelyBalancedProbabilityTransform=DoublingMBPT()):
        self._files = files
        self.balanced_transform = score_transform

    def buildTokeniser(self):
        english_canine_mbr = getEnglishCANINE()
        return BoMMa_Product(
            preprocessor=self._files.preprocessorEffective(),
            max_step=20,

            score_generator=BoundaryScoresChosen(self.balanced_transform),

            vocabulary_constraint_class=VocabularyConstraintExact,
            vocab=self._files.buildVocabulary()
        ).from_object(english_canine_mbr)


class Factory_LeastTokenThenProbability(TokeniserFactory[LeastTokenViterbiWithProbabilityTiebreaker]):
    def __init__(self, files: Deserialiser):
        self._files = files

    def buildTokeniser(self):
        english_canine_mbr = getEnglishCANINE()
        return LeastTokenViterbiWithProbabilityTiebreaker(
            preprocessor=self._files.preprocessorEffective(),
            max_step=20,

            vocab=self._files.buildVocabulary(),
            logprob_classifier=english_canine_mbr
        )


class Factory_ProbabilityThenLeastToken(TokeniserFactory[ProbabilityViterbiWithLeastTokenTiebreaker]):
    def __init__(self, files: Deserialiser):
        self._files = files

    def buildTokeniser(self):
        english_canine_mbr = getEnglishCANINE()
        return ProbabilityViterbiWithLeastTokenTiebreaker(
            preprocessor=self._files.preprocessorEffective(),
            max_step=20,

            vocab=self._files.buildVocabulary(),
            logprob_classifier=english_canine_mbr
        )


class Factory_CanineBPEdropout(TokeniserFactory[GuidedBPEDropout]):

    def __init__(self, files: BPE_Deserialiser=BPE32ki_SlimPajama3M(), deterministic_threshold: float=None):
        self._files = files
        self.threshold = deterministic_threshold

    def buildTokeniser(self):
        english_canine_mbr = getEnglishCANINE()
        return GuidedBPEDropout(
            preprocessor=self._files.preprocessorEffective(),

            vocab=self._files.buildVocabulary(),
            merges=self._files.buildMerges(),

            dropout_probability=english_canine_mbr,
            always_dropout_above=self.threshold,
        )


class Factory_Character(TokeniserFactory[UnicodeTokeniser[WithSpecials]]):
    def __init__(self, specials: WithSpecials):
        self._specials = specials

    def buildTokeniser(self):
        return UnicodeTokeniser(preprocessor=IdentityPreprocessor, specials=self._specials)


class Factory_Switch(TokeniserFactory[StochasticTokeniserSwitch]):

    def __init__(self, global_preprocessor: Preprocessor, use_specific_preprocessors: bool,
                 factory1: TokeniserFactory, factory2: TokeniserFactory, probability_of_second_tk: float=0.5):
        self._glob = global_preprocessor
        self._do_specifics = use_specific_preprocessors
        self._f1 = factory1
        self._f2 = factory2
        self._p = probability_of_second_tk

    def buildTokeniser(self):
        return StochasticTokeniserSwitch(
            preprocessor=MultiplexedPreprocessor(
                global_preprocessor=self._glob,
                specific_preprocessors=self._do_specifics
            ),
            tokeniser1=self._f1.buildTokeniser(),
            tokeniser2=self._f2.buildTokeniser(),
            p=self._p
        )


class Factory_GRaMPa(TokeniserFactory[GRaMPa]):

    def __init__(self, preprocessor: Preprocessor=None, vocab_file: Deserialiser=KudoPiece32ki_SlimPajama3M(),
                 minimal_length: int=1, temperature: float=1.0, r2l_not_l2r: bool=False):
        self._prep = preprocessor
        self._vocab_file = vocab_file
        self._temp = temperature
        self._minlen = minimal_length
        self._r2l = r2l_not_l2r

    def buildTokeniser(self):
        if self._prep is None:
            self._prep = self._vocab_file.preprocessorEffective()  # Effective preprocessor because GRaMPa inference has no built-in preprocessor.

        return GRaMPa(
            preprocessor=self._prep,
            vocab=self._vocab_file.buildVocabulary(),

            probabilities_to_probabilities=PowerNormalisation(temperature=self._temp),
            minimal_token_length=self._minlen,
            decode_backwards=self._r2l
        )


class Factory_SwitchyGrampa_ULM(TokeniserFactory[StochasticTokeniserSwitch]):
    """
    Note: the multiplexer's global preprocessor and the GRaMPa preprocessor are both predetermined to be respectively
    a whitespace+punctuation splitter and a ModernEnglishPreprocessor. If you don't like these defaults, then just use
    the StochasticTokeniserSwitch constructor directly or make your own factory.
    """

    def __init__(self, files: KudoPiece_Deserialiser=KudoPiece32ki_SlimPajama3M(), p: float=0.5,
                 temperature: float=1.0, l_min: int=1,
                 kbest: int=1, smoothing_power: float=1.0):
        self._files = files
        self.p = p

        self.t = temperature
        self.l = l_min

        self.kbest = kbest
        self.smoothing_power = smoothing_power

    def buildTokeniser(self):
        global_preprocessor = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())

        build1 = Factory_KudoPiece(
            files=self._files,
            kbest=self.kbest,
            alpha=self.smoothing_power
        )
        build2 = Factory_GRaMPa(
            preprocessor=ModernEnglishPreprocessor(marker=detectBoundaryMarkerFromVocabulary(self._files.buildVocabulary())),
            vocab_file=self._files,
            minimal_length=self.l,
            temperature=self.t
        )

        return StochasticTokeniserSwitch(
            preprocessor=MultiplexedPreprocessor(
                global_preprocessor=global_preprocessor,
                specific_preprocessors=True
            ),
            tokeniser1=build1.buildTokeniser(),
            tokeniser2=build2.buildTokeniser(),
            p=self.p
        )


class Factory_SwitchyGrampa_BPE(TokeniserFactory[StochasticTokeniserSwitch]):
    """
    Note: the multiplexer's global preprocessor and the BPE/GRaMPa preprocessor are both predetermined to be respectively
    a whitespace+punctuation splitter and a ModernEnglishPreprocessor. If you don't like these defaults, then just use
    the StochasticTokeniserSwitch constructor directly or make your own factory.
    """

    def __init__(self, files: BPE_Deserialiser=BPE32ki_SlimPajama3M(), p: float=0.5,
                 temperature: float=1.0, l_min: int=1,
                 dropout: float=0.0):
        self._files = files
        self.p = p

        self.t = temperature
        self.l = l_min

        self.dropout = dropout

    def buildTokeniser(self):
        global_preprocessor = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())
        sub_preprocessor = ModernEnglishPreprocessor(marker=detectBoundaryMarkerFromVocabulary(self._files.buildVocabulary()))
        build1 = Factory_BPE(
            preprocessor=sub_preprocessor,
            files=self._files,
            dropout=self.dropout
        )
        build2 = Factory_GRaMPa(
            preprocessor=sub_preprocessor,
            vocab_file=self._files,
            minimal_length=self.l,
            temperature=self.t
        )
        return StochasticTokeniserSwitch(
            preprocessor=MultiplexedPreprocessor(
                global_preprocessor=global_preprocessor,
                specific_preprocessors=True
            ),
            tokeniser1=build1.buildTokeniser(),
            tokeniser2=build2.buildTokeniser(),
            p=self.p
        )
