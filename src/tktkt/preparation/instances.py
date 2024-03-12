"""
Some common instantiations of preparation functions.
"""
from ..interfaces.preparation import Preprocessor
from .splitters import *
from .mappers import *

# Most common space markers
SennrichSpaceMarker = SpaceMarker("</w>",    detached=False, location=SpaceMarkerLocation.END)    # Sennrich 2016
RobertaSpaceMarker  = SpaceMarker("Ġ",       detached=True,  location=SpaceMarkerLocation.START)  # Radford 2019
IsolatedSpaceMarker = SpaceMarker("[SPACE]", detached=True,  location=SpaceMarkerLocation.TOKEN)  # Huck 2017

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
            PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.INCLUDED, preserve_spaces_at=marker.location),
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
            PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.EXCLUDED),
            WhitespacePretokeniser(destructive=True),
            MapperAsPretokeniser(PseudoByteMapping()),
            AddWordBoundary(marker),
            IsolateDigits(),
            PunctuationPretokeniser(PunctuationPretokeniser.HyphenMode.ONLY)
        ])

class SemanticPreprocessor(Preprocessor):
    def __init__(self, marker: SpaceMarker):
        super().__init__(HuggingFaceNormaliser(tn.NFKC()), IdentityMapper(), SemanticPretokeniser(marker))


class HuggingFacePreprocessor(Preprocessor):

    def __init__(self, hf_model: PreTrainedTokenizerFast):
        super().__init__(
            uninvertible_mapping=HuggingFaceNormaliser.fromFullTokeniser(hf_model),
            splitter=HuggingFacePretokeniser.fromFullTokeniser(hf_model)
        )


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
            uninvertible_mapping=SequenceMapper([
                HuggingFaceNormaliser.fromFullTokeniser(hf_model),
                Stripper()  # Whatever the HF normaliser does, we want to control all space.
            ]),
            splitter=PretokeniserSequence([
                MapperAsPretokeniser(AppendSpace(front_not_back=True)),  # We know the HF pretokeniser uses spaces as word boundary, so we add it first.
                HuggingFacePretokeniser.fromFullTokeniser(hf_model)
            ])
        )
