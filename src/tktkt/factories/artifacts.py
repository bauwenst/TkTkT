"""
Code that brings specific instances of pre-trained tokenisers into Python data structures.

Essentially, this file replaces HuggingFace's string-based
    AutoTokenizer.from_pretrained(name)
interface with an import- and class-based
    from tktkt.factories.artifacts import Name
    Name()
interface, which has the benefit of being typed and allowing to specify the necessary preprocessor in Python itself
rather than some strange serialised format (unlike e.g. spm.model files).

Perhaps that means these classes should actually live in a separate package entirely (although currently, the
amount of artifacts is limited enough not to bother with this).
"""
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json

from modest.formats.tsv import iterateTsv

from .specials import BertSpecials, RobertaSpecials
from ..interfaces import Vocab, Artifacts
from ..interfaces.identifiers import AutoVocab, NoSpecials, WithSpecials, AutoVocabSpecs
from ..models.predictive.viterbi import HuggingFaceForBinaryCharacterClassification

from ..util.trie import PrefixTrie, SuffixTrie
from ..paths import relativeToCwd, TkTkTPaths
from .preprocessors import *

__all__ = ["BPE40k_Oscar30M_en", "BPE32ki_SlimPajama3M", "BPE50k_RobertaBase", "KudoPiece30k_BooksWiki_en", "KudoPiece32ki_SlimPajama3M"]


Merges = list[tuple[str,...]]

def getEnglishCANINE() -> HuggingFaceForBinaryCharacterClassification:
    # TODO: Eventually, this should become a HF checkpoint.
    PATH_CANINE_FOR_MBR_EN = relativeToCwd(TkTkTPaths.pathToCheckpoints() / "CANINE-C_MBR-en_2024-02-12_19-35-28")

    return HuggingFaceForBinaryCharacterClassification(
        characterclassifier_checkpoint=PATH_CANINE_FOR_MBR_EN.as_posix(),
        input_kwargs={"padding": "max_length", "max_length": 4}  # This is necessary for CANINE because it needs an input of size at least 4. This isn't a problem in fine-tuning because there we're not sending in single examples but 32 at once and collating.
    )


class BPE_Artifacts(Artifacts[WithSpecials]):
    @abstractmethod
    def buildMerges(self) -> Merges:
        pass


class BPE40k_Oscar30M_en(BPE_Artifacts[WithSpecials]):
    """
    Trained with the HuggingFace BPE trainer.

    FIXME: This was never uploaded!
    """
    def getFolder(self) -> Path:
        raise NotImplementedError()

    def _bakedSpecials(self) -> set[str]:
        raise NotImplementedError()

    def _buildVocabulary(self) -> Vocab:  # TODO: Where do I get to choose UNK?
        from bpe_knockout.util.storage import SennrichTokeniserPath
        from ..models.bpe.vocabularisation import BPEVocabulariser
        files = self.getFolder()
        assert isinstance(files, SennrichTokeniserPath)
        return BPEVocabulariser.load(file_or_folder=files.getPaths()[0], specials=self._specials, filtered_types=self._bakedSpecials())

    def buildMerges(self) -> Merges:
        files = self.getFolder()
        return [tuple(m.split(" ")) for m in files.loadMerges()]

    def preprocessorNative(self) -> Preprocessor:
        return Preprocessor(
            splitter=BoundariesFromSpacesPretokeniser(marker=RobertaSpaceMarker, byte_based=True)
        )

    def preprocessorEffective(self) -> Preprocessor:
        return self.preprocessorNative()


class BPE32ki_SlimPajama3M(BPE_Artifacts[WithSpecials]):
    """
    Trained with SentencePiece.

    In the original project, RoBERTa specials were used with IDs
        '<s>': 0
        '</s>': 1
        '<unk>': 2
        '<pad>': 3
        '<mask>': 4
    """
    def _buildVocabulary(self) -> Vocab:
        from ..models.bpe.vocabularisation import BPEVocabulariser
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="vocab.json"))
        return BPEVocabulariser.load(file_or_folder=downloaded_vocab, specials=self._specials)

    def buildMerges(self) -> Merges:
        from ..models.bpe.vocabularisation import BPEVocabulariser
        downloaded_merges = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="merges.txt"))
        return BPEVocabulariser.loadMerges(file_or_folder=downloaded_merges)

    def _bakedSpecials(self) -> set[str]:
        return set()

    def preprocessorNative(self) -> Preprocessor:
        """
        Note: this vocabulary was post-processed from SentencePiece format to HF format, so you can't give this
        vocabulary to SentencePiece anymore. The preprocessor below was used for vocabularisation, so it works
        if you want to do an identical re-vocabularisation for reproduction purposes, but you can't use it for inference.
        """
        return SentencePiecePreprocessor_SpaceConcatenable(marker_location=RobertaSpaceMarker.location)

    def preprocessorEffective(self) -> Preprocessor:
        return ModernEnglishPreprocessor(marker=RobertaSpaceMarker)


class BPE_Artifacts_HuggingFace(BPE_Artifacts[WithSpecials]):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _checkpointName(self) -> str:
        pass

    @abstractmethod
    def _specialsTemplate(self) -> WithSpecials:
        pass

    @abstractmethod
    def _specialsToTypes(self) -> dict[str, str]:
        pass

    def _bakedSpecials(self) -> set[str]:  # Not used for anything because of AutoVocab, but still implementing it properly.
        return set(self._specialsToTypes().values())

    def _buildVocabulary(self) -> Vocab[WithSpecials]:
        return AutoVocab.fromTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self._checkpointName()),
            specials_specification=AutoVocabSpecs(specials_template=self._specialsTemplate(), special_to_string=self._specialsToTypes())
        )

    def buildMerges(self) -> Merges:
        from bpe_knockout.model.auto import AutoMerges
        return AutoMerges.from_pretrained(self._checkpointName())

    def preprocessorNative(self) -> Preprocessor:
        return self.preprocessorEffective()

    def preprocessorEffective(self) -> Preprocessor:  # Note: these tokenisers tend to suck with boundary marking.
        return HuggingFacePreprocessor(AutoTokenizer.from_pretrained(self._checkpointName()))


class BPE50k_RobertaBase(BPE_Artifacts_HuggingFace[RobertaSpecials]):
    """
    The BPE tokeniser for RoBERTa-base.
    """
    def _checkpointName(self) -> str:
        return "FacebookAI/roberta-base"

    def _specialsTemplate(self) -> RobertaSpecials:
        return RobertaSpecials(BOS=0, EOS=0, PAD=0, MASK=0)

    def _specialsToTypes(self) -> dict[str, str]:
        return {
            "BOS": "<s>",
            "EOS": "</s>",
            "PAD": "<pad>",
            "MASK": "<mask>"
        }


class KudoPiece_Artifacts(Artifacts[WithSpecials]):
    @abstractmethod
    def getModelFile(self) -> Path:
        pass

    @abstractmethod
    def loadLikelihoods(self) -> Dict[str, float]:
        pass


class KudoPiece_Artifacts_HuggingFace(KudoPiece_Artifacts[WithSpecials]):
    """
    For vocabularies that were not trained with TkTkT and thus have all their IDs predetermined.
    Uses AutoVocab.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _specialsTemplate(self) -> WithSpecials:
        pass
    
    @abstractmethod
    def _specialsToTypes(self) -> dict[str,str]:
        pass

    @abstractmethod
    def _checkpointName(self) -> str:
        pass

    @abstractmethod
    def _modelFileName(self) -> str:
        pass

    @abstractmethod
    def _jsonFileName(self) -> str:
        pass

    def _bakedSpecials(self) -> set[str]:  # Not used for anything because of AutoVocab, but still implementing it properly.
        return set(self._specialsToTypes().values())

    def _buildVocabulary(self) -> Vocab[WithSpecials]:
        return AutoVocab.fromTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self._checkpointName()),
            specials_specification=AutoVocabSpecs(specials_template=self._specialsTemplate(), special_to_string=self._specialsToTypes())
        )

    def loadLikelihoods(self) -> Dict[str, float]:
        tokeniser_path = Path(hf_hub_download(repo_id=self._checkpointName(), filename=self._jsonFileName()))

        out = dict()
        with open(tokeniser_path, "r", encoding="utf-8") as handle:
            d = json.load(handle)
            for typ, prob in d["model"]["vocab"]:
                out[typ] = float(prob)
        return out

    def getModelFile(self) -> Path:
        return Path(hf_hub_download(repo_id=self._checkpointName(), filename=self._modelFileName()))

    def preprocessorNative(self) -> Preprocessor:
        return IdentityPreprocessor  # The tokeniser probably has all its preprocessing baked into the spiece.model.

    def preprocessorEffective(self) -> Preprocessor:
        # We assume they didn't use the pretoken separator trick TkTkT used, and probably just used spaces.
        preprocessor = SentencePiecePreprocessor_SpaceConcatenable(marker_location=KudoSpaceMarker.location, prefix_space_already_added=False)
        preprocessor.splitter = PretokeniserSequence([
            preprocessor.splitter,
            MapperAsPretokeniser(ReplaceBoundary(" ", KudoSpaceMarker))
        ])
        return preprocessor


class KudoPiece30k_BooksWiki_en(KudoPiece_Artifacts_HuggingFace[BertSpecials]):
    def _checkpointName(self) -> str:
        return "albert/albert-base-v2"

    def _modelFileName(self) -> str:
        return "spiece.model"

    def _jsonFileName(self) -> str:
        return "tokenizer.json"

    def _specialsTemplate(self) -> BertSpecials:
        return BertSpecials(CLS=2, SEP=3, PAD=0, MASK=4)

    def _specialsToTypes(self) -> dict[str, str]:
        return {
            "CLS": "[CLS]",
            "SEP": "[SEP]",
            "PAD": "<pad>",
            "MASK": "[MASK]"
        }


class KudoPiece32ki_SlimPajama3M(KudoPiece_Artifacts[WithSpecials]):
    """
    From the same project as BPE32ki_SlimPajama3M, where it had the same specials.
    """
    def getVocabFile(self) -> Path:
        return Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.vocab"))

    def _buildVocabulary(self) -> Vocab:
        from ..models.kudopiece.vocabularisation import KudoPieceVocabulariser
        return KudoPieceVocabulariser.load(file_or_folder=self.getVocabFile(), specials=self._specials, filtered_types=self._bakedSpecials())

    def _bakedSpecials(self) -> set[str]:
        return {"<s>", "</s>", "<unk>"}

    def getModelFile(self) -> Path:
        return Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.model"))

    def loadLikelihoods(self) -> Dict[str, float]:
        return {t: float(l) for t,l in iterateTsv(self.getVocabFile())}

    def preprocessorNative(self) -> Preprocessor:
        return SentencePiecePreprocessor_SpaceConcatenable(marker_location=KudoSpaceMarker.location, prefix_space_already_added=True)  # E.g. say our preprocessor could produce a string "New York", will be sent to the tokeniser as "New York", which will turn it into " New York" and turn that into "_New_York".

    def preprocessorEffective(self) -> Preprocessor:
        preprocessor = SentencePiecePreprocessor_SpaceConcatenable(marker_location=KudoSpaceMarker.location, prefix_space_already_added=False)
        preprocessor.splitter = PretokeniserSequence([
            preprocessor.splitter,
            MapperAsPretokeniser(ReplaceBoundary(" ", KudoSpaceMarker))
        ])
        return preprocessor


########################################################################################################################


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
