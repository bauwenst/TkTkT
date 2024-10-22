from tktkt.preparation.instances import RobertaSpaceMarker
from tktkt.visualisation.bpe.trees import BpeVisualiser
from tktkt.models.bpe.base import ClassicBPE
from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath
from bpe_knockout.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯, setupEnglish, KnockoutDataConfiguration
from transformers import AutoTokenizer


def minimalWorkingExample():
    from tktkt.models.bpe.base import ClassicBPE
    from tktkt.visualisation.bpe.trees import BpeVisualiser

    bpe = ClassicBPE.fromHuggingFace(AutoTokenizer.from_pretrained("roberta-base"))
    viz = BpeVisualiser(bpe)
    tokens, latex = viz.prepareAndTokenise_visualised_animated("horseshoe")

    print(viz.getLatexPreamble())
    print(latex)
    print(tokens)


def makeBPE1():
    """
    This one produces the iconic horses/hoe.
    """
    with KnockoutDataConfiguration(setupEnglish()):
        vocab_and_merges = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser
        # vocab_and_merges = HuggingFaceTokeniserPath.fromName("roberta-base")
        print("Loading BPE tokeniser from", vocab_and_merges.path)

    bpe = ClassicBPE(
        preprocessor=None,
        vocab=vocab_and_merges.loadVocabulary(),
        merges=vocab_and_merges.loadMerges(),

        boundary_marker=RobertaSpaceMarker
    )
    return bpe


def makeBPE2():
    return ClassicBPE.fromHuggingFace(AutoTokenizer.from_pretrained("roberta-base"))


def makeBPE3():
    bpe = ClassicBPE.from_pretrained_tktkt(checkpoint="Bauwens/RoBERTa-nl_BPE-knockout_30k")
    return bpe


if __name__ == "__main__":
    bpe = makeBPE1()
    viz = BpeVisualiser(bpe)
    tokens, latex = viz.prepareAndTokenise_visualised_animated("horseshoe")
    print(latex)
    print(tokens)
