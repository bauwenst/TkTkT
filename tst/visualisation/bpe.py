from tktkt.preparation.instances import RobertaSpaceMarker
from tktkt.visualisation.bpe.trees import BpeVisualiser, BTE
from tktkt.models.bpe.base import ClassicBPE
from bpe_knockout.auxiliary.tokenizer_interface import HuggingFaceTokeniserPath
from bpe_knockout.project.config import Pℛ𝒪𝒥ℰ𝒞𝒯, setupEnglish, KnockoutDataConfiguration

with KnockoutDataConfiguration(setupEnglish()):
    vocab_and_merges = Pℛ𝒪𝒥ℰ𝒞𝒯.config.base_tokeniser
    # vocab_and_merges = HuggingFaceTokeniserPath.fromName("roberta-base")

bpe = ClassicBPE(
    vocab_and_merges.loadVocabulary(),
    vocab_and_merges.loadMerges(),

    boundary_marker=RobertaSpaceMarker
)

print(vocab_and_merges.path)
viz = BpeVisualiser(bpe)
tokens, latex = viz.prepareAndTokenise_visualised_animated("horseshoe")
print(latex)
print(tokens)
