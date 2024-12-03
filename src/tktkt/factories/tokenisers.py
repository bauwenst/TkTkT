"""
Evaluate any tokeniser on English morphology.
"""
from ..interfaces.factories import TokeniserFactory, T
from ..factories.deserialisation import *
from ..preparation.instances import *
from ..models.viterbi.instances import *
from ..models.bpe.base import ClassicBPE
from ..models.bpe.knockout import BPEKnockout, ReBPE
from ..models.bpe.guided import GuidedBPEDropout
from ..models.huggingface.wrapper import HuggingFaceTokeniser
from ..models.huggingface.bpe import HuggingFaceBPETokeniser
from ..models.kudopiece.segmentation import KudoPieceTokeniser
from ..models.random.pathmarkov import GRaMPa, PowerNormalisation
from ..models.ngram.alphabet import UnicodeTokeniser
from ..wrappers.multiplexing import StochasticTokeniserSwitch, MultiplexedPreprocessor


########################################################################################################################


class Factory_BPE(TokeniserFactory[HuggingFaceTokeniser]):
    def __init__(self, preprocessor: Preprocessor=None, dropout: float=0.0, files: BPE_Deserialiser=BPE40k_Oscar30M_en()):
        self._prep = preprocessor
        self._dropout = dropout
        self._files = files

    def buildTokeniser(self):
        vocab = self._files.buildVocabulary()
        if self._prep is None:
            self._prep = self._files.preprocessor()

        merges = self._files.buildMerges()
        return HuggingFaceBPETokeniser(vocab, merges, dropout=self._dropout, preprocessor=self._prep)
        # english_bpe = getEnglishBpeFiles().toFastBPE()  # HuggingFace automatically sets a ByteBased tokenizers.pretokeniser on all RobertaTokenizerFast instances, which also implicitly adds a start-of-word Ġ as replacement for spaces.
        # return HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)


class Factory_BPE_native(TokeniserFactory[ClassicBPE]):
    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return ClassicBPE(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            boundary_marker=RobertaSpaceMarker
        )


class Factory_BPEKnockout(TokeniserFactory[BPEKnockout]):
    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return BPEKnockout(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker
        )


class Factory_ReBPE(TokeniserFactory[ReBPE]):
    def __init__(self, iterations: int, reduced: bool=False):
        self.its = iterations
        self.bc = reduced

    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return ReBPE(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker,

            iterations=self.its,
            backwards_compatible=self.bc
        )


class Factory_KudoPiece(TokeniserFactory[KudoPieceTokeniser]):
    """
    Defaults to the 32k SlimPajama vocab.
    """

    def __init__(self, preprocessor: Preprocessor=None, files: KudoPiece_Deserialiser=KudoPiece32ki_SlimPajama3M(), kbest: int=64, alpha: float=1.0):
        self._prep = preprocessor
        self._files = files
        self._kbest = kbest
        self._alpha = alpha

    def buildTokeniser(self):
        vocab = self._files.buildVocabulary()
        if self._prep is None:
            self._prep = self._files.preprocessorForSentencePieceInference()

        return KudoPieceTokeniser(
            preprocessor=self._prep,
            model_file=self._files.getModelFile(),
            vocab=vocab,

            kbest=self._kbest,
            smoothing_power=self._alpha
        )


class Factory_LeastToken_BPE(TokeniserFactory[LeastTokenViterbi]):
    def buildTokeniser(self) -> T:
        english_bpe = getEnglishBpeFiles().toFastBPE()
        return LeastTokenViterbi(
            preprocessor=HuggingFacePreprocessorForWords(english_bpe),
            vocab=english_bpe.get_vocab(),
            max_step=20
        )


class Factory_LeastToken_BPEKnockout(TokeniserFactory[LeastTokenViterbi]):
    def buildTokeniser(self) -> T:
        # Get starting BPE vocabulary
        files = getEnglishBpeFiles()

        # Prune the vocabulary with BPE-knockout
        only_for_vocabulary = BPEKnockout(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),
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


class Factory_LeastToken_ULM(TokeniserFactory[LeastTokenViterbi]):
    def buildTokeniser(self) -> T:
        hf_english_ulm = getEnglishKudo()
        return LeastTokenViterbi(
            HuggingFacePreprocessorForWords(hf_english_ulm),
            vocab=hf_english_ulm.get_vocab(),
            max_step=20
        )


class Factory_BoMMaSum_BPE(TokeniserFactory[BoMMa_Sum]):
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


class Factory_BoMMaSum_ULM(TokeniserFactory[BoMMa_Sum]):
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


class Factory_BoMMaSum_FromTransform_ULM(Factory_BoMMaSum_ULM):
    def __init__(self,
        generator: Type[ScoreGeneratorUsingCharacterClassifierForTransform]=BoundaryScoresChosen,
        score_transform: ProbabilityTransform=LinearPT(-1, +1, negate_as_complement=False),
        constraint: Type[VocabularyConstraint]=VocabularyConstraintExact
    ):
        super().__init__(generator(score_transform), constraint)


class Factory_BoMMaProduct_ULM(TokeniserFactory[BoMMa_Product]):
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


class Factory_LeastTokenThenProbability_ULM(TokeniserFactory[LeastTokenViterbiWithProbabilityTiebreaker]):
    def buildTokeniser(self) -> T:
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()
        return LeastTokenViterbiWithProbabilityTiebreaker(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            max_step=20,

            vocab=english_ulm.get_vocab(),
            logprob_classifier=english_canine_mbr
        )


class Factory_ProbabilityThenLeastToken_ULM(TokeniserFactory[ProbabilityViterbiWithLeastTokenTiebreaker]):
    def buildTokeniser(self) -> T:
        english_ulm        = getEnglishKudo()
        english_canine_mbr = getEnglishCANINE()
        return ProbabilityViterbiWithLeastTokenTiebreaker(
            preprocessor=HuggingFacePreprocessorForWords(english_ulm),
            max_step=20,

            vocab=english_ulm.get_vocab(),
            logprob_classifier=english_canine_mbr
        )


class Factory_CanineBPEdropout(TokeniserFactory[GuidedBPEDropout]):

    def __init__(self, deterministic_threshold: float=None):
        self.threshold = deterministic_threshold

    def buildTokeniser(self) -> T:
        english_bpe_files  = getEnglishBpeFiles()
        english_canine_mbr = getEnglishCANINE()
        return GuidedBPEDropout(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # We know the BPE files uses this marker, so we can manually specify it.

            vocab=english_bpe_files.loadVocabulary(),
            merges=english_bpe_files.loadMerges(),
            boundary_marker=RobertaSpaceMarker,

            dropout_probability=english_canine_mbr,
            always_dropout_above=self.threshold,
        )


class Factory_Character(TokeniserFactory[UnicodeTokeniser]):
    def buildTokeniser(self) -> T:
        return UnicodeTokeniser(preprocessor=IdentityPreprocessor)


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

    def __init__(self, preprocessor: Preprocessor, vocab_file: Deserialiser, minimal_length: int, temperature: float, r2l_not_l2r: bool=False):
        self._prep = preprocessor
        self._vocab_file = vocab_file
        self._temp = temperature
        self._minlen = minimal_length
        self._r2l = r2l_not_l2r

    def buildTokeniser(self):
        return GRaMPa(
            preprocessor=self._prep,
            vocab=self._vocab_file.buildVocabulary(),
            # unk_type=self._vocab._specials.unk_token,

            probabilities_to_probabilities=PowerNormalisation(temperature=self._temp),
            minimal_token_length=self._minlen,
            decode_backwards=self._r2l
        )


class Factory_SwitchyGrampa_ULM(TokeniserFactory[StochasticTokeniserSwitch]):

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
