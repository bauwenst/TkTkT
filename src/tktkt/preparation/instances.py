"""
Some common instantiations of preparation functions.
"""
from ..interfaces.preparation import Preprocessor
from .splitters import *
from .mappers import *
from .huggingface import *

# Most common space markers
SennrichSpaceMarker = SpaceMarker("</w>",    detached=False, location=SpaceMarkerLocation.END)       # Sennrich 2016
RobertaSpaceMarker  = SpaceMarker("Ġ",       detached=True,  location=SpaceMarkerLocation.START)     # Radford 2019
IsolatedSpaceMarker = SpaceMarker("[SPACE]", detached=True,  location=SpaceMarkerLocation.ISOLATED)  # Huck 2017

IdentityPreprocessor = Preprocessor(IdentityMapper(), IdentityMapper(), IdentityPretokeniser())

TraditionalPretokeniser = PretokeniserSequence([
    WhitespacePretokeniser(destructive=True),
    PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.EXCLUDED)
])

class BoundariesFromSpacesPretokeniser(PretokeniserSequence):
    """
    Generalisation of the RoBERTa/GPT-2 tokeniser. Principles it holds by:
        - Punctuation is always isolated from non-punctuation.
        - Word boundaries are only placed where there was a space originally.

    The original implementation uses a start-of-word Ġ and is byte-based. For example, a sentence
        This is a (test) sentënce.
    becomes
        ĠThis // Ġis // Ġa // Ġ( // test // ) // ĠsentÃ«nce // .

    My implementation allows customising the boundary marker and turning off the byte-based behaviour.
    """
    def __init__(self, marker: SpaceMarker, byte_based: bool):
        super().__init__([
            WhitespacePretokeniser(destructive=True),
            AddWordBoundary(SpaceMarker(" ", detached=True, location=marker.location)),
            PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.INCLUDED, group_adjacent_spaces_with_punctuation=marker.location),
            (MapperAsPretokeniser(PseudoByteMapping()) if byte_based else MapperAsPretokeniser(Replace(" ", "Ġ"))),
            MapperAsPretokeniser(Replace("Ġ", " ")),
            WhitespaceAndMarkerPretokeniser(marker)
        ])

# There is only one difference between the original Roberta pretokeniser and this one, which is that multiple spaces are (surprisingly) conserved in the original.
RobertaPretokeniser = BoundariesFromSpacesPretokeniser(marker=RobertaSpaceMarker, byte_based=True)

# The following, although it makes intuitive sense, doesn't conform to the original Roberta pretokeniser because 1. it should split off punctuation (but not from the SoW!) and 2. it auto-adds a space at the start.
# RobertaPretokeniser = PretokeniserSequence([
#     WhitespacePretokeniser(destructive=True),
#     MapperAsPretokeniser(PseudoByteMapping()),
#     AddSpaceMarker(RobertaSpaceMarker)
# ])
RobertaPreprocessor = Preprocessor(IdentityMapper(), IdentityMapper(), RobertaPretokeniser)

class SemanticPretokeniser(PretokeniserSequence):
    """
    My common-sense pretokeniser can add any space marker for announcing words/punctuations.
    """
    def __init__(self, marker: SpaceMarker):
        super().__init__([
            PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
            WhitespacePretokeniser(destructive=True),
            MapperAsPretokeniser(PseudoByteMapping()),
            AddWordBoundary(marker),
            IsolateDigits(),
            PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.ONLY),
            EnglishApostrophes(do_nt=True)
        ])

class SemanticPreprocessor(Preprocessor):
    def __init__(self, marker: SpaceMarker):
        super().__init__(HuggingFaceNormaliser(tn.NFKC()), IdentityMapper(), SemanticPretokeniser(marker))
