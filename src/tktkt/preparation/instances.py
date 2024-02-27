"""
Some common instantiations of preparation functions.
"""
from ..interfaces.preparation import Preprocessor
from .splitters import *
from .mappers import *

SennrichSpaceMarker = SpaceMarker("</w>",    detached=False, location=SpaceMarkerLocation.END)
RobertaSpaceMarker  = SpaceMarker("Ġ",       detached=True,  location=SpaceMarkerLocation.START)
IsolatedSpaceMarker = SpaceMarker("[SPACE]", detached=True,  location=SpaceMarkerLocation.TOKEN)

# There is only one difference between the original Roberta pretokeniser and this one, which is that multiple spaces are (surprisingly) conserved in the original.
RobertaPretokeniser = PretokeniserSequence([
    PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.INCLUDED, grab_preceding_space=True),
    WhitespaceAndMarkerPretokeniser(SpaceMarker(" ", detached=True, location=SpaceMarkerLocation.START)),  # Split on spaces and add the space as a prefix.
    MapperAsPretokeniser(PseudoByteMapping())  # Converts the above space into a Ġ and does other byte mappings. Indeed, we do not make any reference to RobertaSpaceMarker, in the philosophy that HuggingFace actually doesn't use a SoW, but just does byte mapping on all characters.
])
# The doesn't conform to the original Roberta pretokeniser because 1. it should split off punctuation (but not from the SoW!) and 2. it auto-adds a space at the start.
# RobertaPretokeniser = PretokeniserSequence([
#     WhitespacePretokeniser(destructive=True),
#     MapperAsPretokeniser(PseudoByteMapping()),
#     AddSpaceMarker(RobertaSpaceMarker)
# ])
RobertaPreprocessor = Preprocessor(IdentityMapper(), IdentityMapper(), RobertaPretokeniser)

# My common-sense pretokeniser can add any space marker for announcing words/punctuations.
CommonsensePretokeniser = PretokeniserSequence([
    PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.EXCLUDED),
    WhitespacePretokeniser(destructive=True),
    MapperAsPretokeniser(PseudoByteMapping()),
    AddWordBoundary(RobertaSpaceMarker),
    IsolateDigits(),
    PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.ONLY)
])
CommonsensePreprocessor = Preprocessor(Normaliser(tn.NFKC()), IdentityMapper(), CommonsensePretokeniser)


class HuggingFacePreprocessorForWords(Preprocessor):
    """
    For tokenising text as if it was an isolated word (i.e. the start of the word is the start of the input and the end
    is the end of the input). Not trivial since by default, tokenisers like the RobertaTokenizerFast assume a string is
    explicitly not at the start of a word if there is no start-of-word marker.

    The motivation behind this implementation:
        - In our framework, adding markers like a start-of-word is straight-forward: you add them to everything after
          splitting on spaces and punctuation, with the idea being that a word should always be started with a SoW (or
          ended with an EoW) regardless of the spaces that surround it. You don't encode text, you encode semantics.
        - We don't, however, have control over the SoW/EoW behaviour of HuggingFace tokenisers, and they DO make
          use of spaces to decide whether to put down a SoW, we are forced to add a space to the input.
        - Since we don't want the user to have to change their input depending on which tokeniser they use, we give them
          this preprocessor to wrap their tokeniser in and hence it will behave consistently.
    """

    def __init__(self, hf_model: PreTrainedTokenizerFast):
        super().__init__(
            uninvertible_mapping=Stripper(),
            splitter=PretokeniserSequence([
                MapperAsPretokeniser(AppendSpace(front_not_back=True)),
                HuggingFacePretokeniser(hf_model)
            ])
        )
