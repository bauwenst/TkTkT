from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, List, Dict

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast

from bpe_knockout.project.config import KnockoutDataConfiguration, setupEnglish, defaultTokeniserFiles
from bpe_knockout.auxiliary.tokenizer_interface import BpeTokeniserPath, SennrichTokeniserPath

from ..interfaces.vocabulariser import Vocab, DEFAULT_FIVE_SPECIALS
from ..models.bpe.vocabularisation import BPEVocabulariser, Merges
from ..models.kudopiece.vocabularisation import KudoPieceTrainer
from ..models.viterbi import HuggingFaceForBinaryCharacterClassification
from ..preparation.boundaries import BoundaryMarker, BoundaryMarkerLocation
from ..util.trie import PrefixTrie, SuffixTrie
from .paths import relativeToCwd, TkTkTPaths


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


class VocabularyLoader(ABC):
    """
    Object that loads a specific instance of vocabularisation results stored to disk.

    Note that it is vocabularisers that already have the knowledge of how to store to disk (and therefore also how to
    load from disk), except when the file format exists somewhere out there but TkTkT doesn't support that training paradigm.

    Vocabularisers represent an algorithm and file format, e.g. BPE in Sennrich format.
    VocabularyLoaders represent the result of an algorithm and file content, e.g. the BPE vocab and merges resulting from
    applying BPE with |V| = 32k to SlimPajama's first 3M examples.
    """

    def __init__(self, specials: Union[Vocab,List[str]]=DEFAULT_FIVE_SPECIALS.all_special_tokens):
        self._specials = specials
        self._vocab_cache = None

    def buildVocabulary(self) -> Vocab:
        if self._vocab_cache is None:
            self._vocab_cache = self._buildVocabulary()
        return self._vocab_cache

    @abstractmethod
    def _buildVocabulary(self) -> Vocab:
        pass


class BPE_VocabLoader(VocabularyLoader):
    @abstractmethod
    def buildMerges(self) -> Merges:
        pass


class Vocab_BPE40k_Oscar30M_en(BPE_VocabLoader):
    def _buildVocabulary(self) -> Vocab:
        files = getEnglishBpeFiles()
        assert isinstance(files, SennrichTokeniserPath)
        return BPEVocabulariser.load(file_or_folder=files.getPaths()[0], existing_types=self._specials)

    def buildMerges(self) -> Merges:
        files = getEnglishBpeFiles()
        return [tuple(m.split(" ")) for m in files.loadMerges()]


class Vocab_BPE32ki_SlimPajama3M(BPE_VocabLoader):
    def _buildVocabulary(self) -> Vocab:
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="vocab.json"))
        return BPEVocabulariser.load(file_or_folder=downloaded_vocab, existing_types=self._specials)

    def buildMerges(self) -> Merges:
        downloaded_merges = Path(hf_hub_download(repo_id="Bauwens/BPE-32k_SlimPajama-3M", filename="merges.txt"))
        return BPEVocabulariser.loadMerges(file_or_folder=downloaded_merges)


class KudoPiece_VocabLoader(VocabularyLoader):
    @abstractmethod
    def getVocabFile(self) -> Path:
        pass

    def loadProbabilities(self) -> Dict[str,float]:
        out = dict()
        with open(self.getVocabFile(), "r", encoding="utf-8") as handle:
            for line in handle:
                typ,prob = line.rstrip().split("\t")
                out[typ] = float(prob)
        return out


class Vocab_KudoPiece30k_BooksWiki_en(KudoPiece_VocabLoader):
    def _buildVocabulary(self) -> Vocab:
        return AutoTokenizer.from_pretrained("albert/albert-base-v2").get_vocab()

    def getVocabFile(self) -> Path:
        return Path(hf_hub_download(repo_id="albert/albert-base-v2", filename="spiece.model"))


class Vocab_KudoPiece32ki_SlimPajama3M(KudoPiece_VocabLoader):
    def _buildVocabulary(self) -> Vocab:
        downloaded_vocab = Path(hf_hub_download(repo_id="Bauwens/ULM-32k_SlimPajama-3M", filename="spm.vocab"))
        return KudoPieceTrainer.load(file_or_folder=downloaded_vocab, existing_types=self._specials)

    def getVocabFile(self) -> Path:
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
