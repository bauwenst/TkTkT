from pathlib import Path
from abc import abstractmethod
from typing import Dict

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast
import json

from bpe_knockout.project.config import KnockoutDataConfiguration, setupEnglish, defaultTokeniserFiles
from bpe_knockout.auxiliary.tokenizer_interface import BpeTokeniserPath, SennrichTokeniserPath

from ..interfaces.vocabulariser import Vocab
from ..interfaces.factories import Deserialiser
from ..models.bpe.vocabularisation import BPEVocabulariser, Merges
from ..models.kudopiece.vocabularisation import KudoPieceVocabulariser
from ..models.viterbi import HuggingFaceForBinaryCharacterClassification
from ..preparation.instances import *
from ..util.trie import PrefixTrie, SuffixTrie
from ..paths import relativeToCwd, TkTkTPaths


# TODO: Eventually, this should become a HF checkpoint.
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


class BPE_Deserialiser(Deserialiser):
    @abstractmethod
    def buildMerges(self) -> Merges:
        pass


class BPE40k_Oscar30M_en(BPE_Deserialiser):
    def _buildVocabulary(self) -> Vocab:
        files = getEnglishBpeFiles()
        assert isinstance(files, SennrichTokeniserPath)
        return BPEVocabulariser.load(file_or_folder=files.getPaths()[0], existing_types=self._specials)

    def buildMerges(self) -> Merges:
        files = getEnglishBpeFiles()
        return [tuple(m.split(" ")) for m in files.loadMerges()]

    def preprocessor(self) -> Preprocessor:
        return Preprocessor(
            splitter=BoundariesFromSpacesPretokeniser(marker=RobertaSpaceMarker, byte_based=True)
        )


class BPE32ki_SlimPajama3M(BPE_Deserialiser):
    def _buildVocabulary(self) -> Vocab:
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="vocab.json"))
        return BPEVocabulariser.load(file_or_folder=downloaded_vocab, existing_types=self._specials)

    def buildMerges(self) -> Merges:
        downloaded_merges = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="merges.txt"))
        return BPEVocabulariser.loadMerges(file_or_folder=downloaded_merges)

    def preprocessor(self) -> Preprocessor:
        return ModernEnglishPreprocessor(marker=RobertaSpaceMarker)


class KudoPiece_Deserialiser(Deserialiser):
    @abstractmethod
    def getModelFile(self) -> Path:
        pass

    @abstractmethod
    def loadProbabilities(self) -> Dict[str,float]:
        pass

    @abstractmethod
    def preprocessorForSentencePieceInference(self) -> Preprocessor:
        """
        The preprocessor to use when you are using this vocabulary's SentencePiece binary.
        For example, if that binary adds a space to every input, then you should not add a space like you otherwise would.
        """
        pass


class KudoPiece30k_BooksWiki_en(KudoPiece_Deserialiser):
    def _buildVocabulary(self) -> Vocab:
        return AutoTokenizer.from_pretrained("albert/albert-base-v2").get_vocab()

    def loadProbabilities(self) -> Dict[str,float]:
        tokeniser_path = Path(hf_hub_download(repo_id="albert/albert-base-v2", filename="tokenizer.json"))

        out = dict()
        with open(tokeniser_path, "r", encoding="utf-8") as handle:
            d = json.load(handle)
            for typ, prob in d["model"]["vocab"]:
                out[typ] = float(prob)
        return out

    def getModelFile(self) -> Path:
        return Path(hf_hub_download(repo_id="albert/albert-base-v2", filename="spiece.model"))

    def preprocessorForSentencePieceInference(self) -> Preprocessor:
        return IdentityPreprocessor  # The Albert tokeniser probably has all its preprocessing baked into the spiece.model.

    def preprocessor(self) -> Preprocessor:
        preprocessor = SentencePiecePreprocessor_SpaceConcatenable(marker_location=KudoSpaceMarker.location, prefix_space_already_added=False)
        preprocessor.splitter = PretokeniserSequence([
            preprocessor.splitter,
            MapperAsPretokeniser(ReplaceBoundary(" ", KudoSpaceMarker))
        ])
        return preprocessor


class KudoPiece32ki_SlimPajama3M(KudoPiece_Deserialiser):
    def _buildVocabulary(self) -> Vocab:
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.vocab"))
        return KudoPieceVocabulariser.load(file_or_folder=downloaded_vocab, existing_types=self._specials)

    def getModelFile(self) -> Path:
        return Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.model"))

    def preprocessorForSentencePieceInference(self) -> Preprocessor:
        return SentencePiecePreprocessor_SpaceConcatenable(marker_location=KudoSpaceMarker.location, prefix_space_already_added=True)  # E.g. say our preprocessor could produce a string "New York", will be sent to the tokeniser as "New York", which will turn it into " New York" and turn that into "_New_York".

    def preprocessor(self) -> Preprocessor:
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
