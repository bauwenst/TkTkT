"""
Evaluate any tokeniser on English morphology.
"""
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from bpe_knockout.project.config import KnockoutDataConfiguration, setupEnglish, defaultTokeniserFiles
from bpe_knockout.auxiliary.tokenizer_interface import BpeTokeniserPath, SennrichTokeniserPath

from .base import TokeniserBuilder, VocabularyBuilder, T, A
from ..preparation.instances import *
from ..models.viterbi.instances import *
from ..models.bpe.vocabularisation import BPEVocabulariser, Merges
from ..models.bpe.base import ClassicBPE
from ..models.bpe.knockout import BPEKnockout, ReBPE
from ..models.bpe.guided import GuidedBPEDropout
from ..models.huggingface.wrapper import HuggingFaceTokeniser
from ..models.huggingface.bpe import HuggingFaceBPETokeniser
from ..models.kudopiece.vocabularisation import KudoPieceTrainer
from ..models.kudopiece.segmentation import KudoPieceTokeniser
from ..models.random.pathmarkov import GRaMPa, PowerNormalisation
from ..models.ngram.alphabet import UnicodeTokeniser
from ..files.paths import relativeToCwd, TkTkTPaths
from ..wrappers.multiplexing import StochasticTokeniserSwitch, MultiplexedPreprocessor
from ..util.trie import PrefixTrie, SuffixTrie


# TODO: Eventually, this should become a HF checkpoint.
PATH_CANINE_FOR_MBR_EN = relativeToCwd(TkTkTPaths.pathToCheckpoints() / "CANINE-C_MBR-en_2024-02-12_19-35-28")


# Helpers
#########

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


class Builder_Vocab_BPE(VocabularyBuilder[Merges]):
    pass


class Vocab_BPE40k_Oscar30M_en(Builder_Vocab_BPE):
    def buildVocabulary(self) -> Vocab:
        files = getEnglishBpeFiles()
        assert isinstance(files, SennrichTokeniserPath)
        return BPEVocabulariser.load(file_or_folder=files.getPaths()[0], existing_types=self._specials)

    def buildAdditionals(self):
        files = getEnglishBpeFiles()
        return files.loadMerges()


class Vocab_BPE32ki_SlimPajama3M(Builder_Vocab_BPE):
    def buildVocabulary(self) -> Vocab:
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="vocab.json"))
        return BPEVocabulariser.load(file_or_folder=downloaded_vocab, existing_types=self._specials)

    def buildAdditionals(self) -> A:
        downloaded_merges = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="merges.txt"))
        return BPEVocabulariser.loadMerges(file_or_folder=downloaded_merges)


class Builder_Vocab_KudoPiece(VocabularyBuilder[Path]):  # If you want to use HuggingFace, you'd want a Dict[str,float].
    pass


class Vocab_KudoPiece30k_BooksWiki_en(Builder_Vocab_KudoPiece):
    def buildVocabulary(self):
        return AutoTokenizer.from_pretrained("albert/albert-base-v2").get_vocab()

    def buildAdditionals(self):
        return Path(hf_hub_download(repo_id="albert/albert-base-v2", filename="spiece.model"))


class Vocab_KudoPiece32ki_SlimPajama3M(Builder_Vocab_KudoPiece):
    def buildVocabulary(self):
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.vocab"))
        return KudoPieceTrainer.load(file_or_folder=downloaded_vocab, existing_types=self._specials)

    def buildAdditionals(self):
        return Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.model"))


def detectBoundaryMarkerFromVocabulary(vocab: Vocab, threshold: float=0.5) -> BoundaryMarker:
    V = len(vocab)

    trie = PrefixTrie()
    for t in vocab:
        trie.add(t)
    trie.compileRoots()

    suggested_prefix = trie
    while True:
        top = suggested_prefix.getTopChildNodes(n=1)
        if len(top) and top[0].count / V >= threshold:
            suggested_prefix = top[0]
        else:
            break

    trie = SuffixTrie()
    for t in vocab:
        trie.add(t)
    trie.compileRoots()

    suggested_suffix = trie
    while True:
        top = suggested_suffix.getTopChildNodes(n=1)
        if len(top) and top[0].count / V >= threshold:
            suggested_suffix = top[0]
        else:
            break

    found_prefix = len(suggested_prefix.root) > 0
    found_suffix = len(suggested_suffix.root) > 0
    prefix = BoundaryMarker(suggested_prefix.root, detached=suggested_prefix.root in vocab, location=BoundaryMarkerLocation.START)
    suffix = BoundaryMarker(suggested_suffix.root, detached=suggested_suffix.root in vocab, location=BoundaryMarkerLocation.END)

    if not found_prefix and not found_suffix:  # No prefix nor suffix? I guess it must be an isolated token.
        return BoundaryMarker("", detached=True, location=BoundaryMarkerLocation.ISOLATED)
    elif not found_suffix:  # No suffix? Then it's a prefix.
        return prefix
    elif not found_prefix:  # No prefix? Then it's a suffix.
        return suffix
    else:  # Prefix and suffix found? Then take the one with higher occurrence. (TODO: Alternatively, take the one with higher length.)
        if suggested_prefix.count > suggested_suffix.count:
            return prefix
        else:
            return suffix


########################################################################################################################


class Builder_English_BPE(TokeniserBuilder[HuggingFaceTokeniser]):
    def __init__(self, preprocessor: Preprocessor=None, dropout: float=0.0, vocab: Builder_Vocab_BPE=Vocab_BPE40k_Oscar30M_en()):
        if preprocessor is None:
            preprocessor = ModernEnglishPreprocessor(marker=detectBoundaryMarkerFromVocabulary(vocab.buildVocabulary()))

        self._prep = preprocessor
        self._dropout = dropout
        self._vocab_builder = vocab

    def buildTokeniser(self):
        vocab, merges = self._vocab_builder.buildVocabulary()
        return HuggingFaceBPETokeniser(vocab, merges, dropout=self._dropout, preprocessor=self._prep)
        # english_bpe = getEnglishBpeFiles().toFastBPE()  # HuggingFace automatically sets a ByteBased tokenizers.pretokeniser on all RobertaTokenizerFast instances, which also implicitly adds a start-of-word Ġ as replacement for spaces.
        # return HuggingFaceTokeniser(wrapped_tokeniser=english_bpe, for_single_words=True)


class Builder_English_BPE_native(TokeniserBuilder[ClassicBPE]):
    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return ClassicBPE(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            boundary_marker=RobertaSpaceMarker
        )


class Builder_English_BPEKnockout(TokeniserBuilder[BPEKnockout]):
    def buildTokeniser(self) -> T:
        files = getEnglishBpeFiles()
        return BPEKnockout(
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
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
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # I use this because I know the BPE vocabs are byte-based and this one is too.
            vocab=files.loadVocabulary(),
            merges=files.loadMerges(),
            language="English",
            boundary_marker=RobertaSpaceMarker,

            iterations=self.its,
            backwards_compatible=self.bc
        )


class Builder_English_KudoPiece(TokeniserBuilder[KudoPieceTokeniser]):
    """
    Defaults to the 32k SlimPajama vocab.
    """

    def __init__(self, preprocessor: Preprocessor=None, vocab: Builder_Vocab_KudoPiece=Vocab_KudoPiece32ki_SlimPajama3M(), kbest: int=64, alpha: float=1.0):
        if preprocessor is None:
            preprocessor = SentencePiecePreprocessor(marker=detectBoundaryMarkerFromVocabulary(vocab.buildVocabulary()), prefix_space_already_added=True)  # Marker is only used for its location. I fucked up and set add_prefix to True when training the tokeniser, and now that option is baked into the .model file LMAO.

        self._prep = preprocessor
        self._vocab = vocab
        self._kbest = kbest
        self._alpha = alpha

    def buildTokeniser(self):
        return KudoPieceTokeniser(
            preprocessor=self._prep,
            model_file=self._vocab.buildAdditionals(),
            vocab=self._vocab.buildVocabulary(),

            kbest=self._kbest,
            smoothing_power=self._alpha
        )


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
            preprocessor=ModernEnglishPreprocessor(RobertaSpaceMarker),  # We know the BPE files uses this marker, so we can manually specify it.

            vocab=english_bpe_files.loadVocabulary(),
            merges=english_bpe_files.loadMerges(),
            boundary_marker=RobertaSpaceMarker,

            dropout_probability=english_canine_mbr,
            always_dropout_above=self.threshold,
        )


class Builder_English_Character(TokeniserBuilder[UnicodeTokeniser]):
    def buildTokeniser(self) -> T:
        return UnicodeTokeniser(preprocessor=IdentityPreprocessor)


class Builder_Switch(TokeniserBuilder[StochasticTokeniserSwitch]):

    def __init__(self, global_preprocessor: Preprocessor, use_specific_preprocessors: bool,
                 builder1: TokeniserBuilder, builder2: TokeniserBuilder, probability_of_second_tk: float=0.5):
        self._glob = global_preprocessor
        self._do_specifics = use_specific_preprocessors
        self._b1 = builder1
        self._b2 = builder2
        self._p = probability_of_second_tk

    def buildTokeniser(self):
        return StochasticTokeniserSwitch(
            preprocessor=MultiplexedPreprocessor(
                global_preprocessor=self._glob,
                specific_preprocessors=self._do_specifics
            ),
            tokeniser1=self._b1.buildTokeniser(),
            tokeniser2=self._b2.buildTokeniser(),
            p=self._p
        )


class Builder_GRaMPa(TokeniserBuilder[GRaMPa]):

    def __init__(self, preprocessor: Preprocessor, vocab: VocabularyBuilder, minimal_length: int, temperature: float):
        self._prep = preprocessor
        self._vocab = vocab
        self._temp = temperature
        self._minlen = minimal_length

    def buildTokeniser(self):
        return GRaMPa(
            preprocessor=self._prep,
            vocab=self._vocab.buildVocabulary(),
            # unk_type=self._vocab._specials.unk_token,

            probabilities_to_probabilities=PowerNormalisation(temperature=self._temp),
            minimal_token_length=self._minlen,
            decode_backwards=False
        )


class Builder_SwitchyGrampa_ULM(TokeniserBuilder[StochasticTokeniserSwitch]):

    def __init__(self, vocab: Builder_Vocab_KudoPiece=Vocab_KudoPiece32ki_SlimPajama3M(), p: float=0.5,
                 temperature: float=1.0, l_min: int=1,
                 kbest: int=1, smoothing_power: float=1.0):
        self.vocab_builder = vocab
        self.p = p

        self.t = temperature
        self.l = l_min

        self.kbest = kbest
        self.smoothing_power = smoothing_power

    def buildTokeniser(self):
        global_preprocessor = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())

        build1 = Builder_English_KudoPiece(
            vocab=self.vocab_builder,
            kbest=self.kbest,
            alpha=self.smoothing_power
        )
        build2 = Builder_GRaMPa(
            preprocessor=ModernEnglishPreprocessor(marker=detectBoundaryMarkerFromVocabulary(self.vocab_builder.buildVocabulary())),
            vocab=self.vocab_builder,
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


class Builder_SwitchyGrampa_BPE(TokeniserBuilder[StochasticTokeniserSwitch]):

    def __init__(self, vocab: Builder_Vocab_BPE=Vocab_BPE32ki_SlimPajama3M(), p: float=0.5,
                 temperature: float=1.0, l_min: int=1,
                 dropout: float=0.0):
        self.vocab_builder = vocab
        self.p = p

        self.t = temperature
        self.l = l_min

        self.dropout = dropout

    def buildTokeniser(self):
        global_preprocessor = Preprocessor(TruncateAndNormalise(1_000_000), IdentityMapper(), TraditionalPretokeniser())
        sub_preprocessor = ModernEnglishPreprocessor(marker=detectBoundaryMarkerFromVocabulary(self.vocab_builder.buildVocabulary()))
        build1 = Builder_English_BPE(
            preprocessor=sub_preprocessor,
            vocab=self.vocab_builder,
            dropout=self.dropout
        )
        build2 = Builder_GRaMPa(
            preprocessor=sub_preprocessor,
            vocab=self.vocab_builder,
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
