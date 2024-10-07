"""
Some common instantiations of preparation functions.
"""
from ..interfaces.preparation import Preprocessor
from .splitters import *
from .mappers import *
from .huggingface import *

# Most common space markers
SennrichSpaceMarker = BoundaryMarker("</w>",    detached=False, location=BoundaryMarkerLocation.END)       # Sennrich 2016
IsolatedSpaceMarker = BoundaryMarker("[SPACE]", detached=True,  location=BoundaryMarkerLocation.ISOLATED)  # Huck 2017
KudoSpaceMarker     = BoundaryMarker("▁",       detached=True,  location=BoundaryMarkerLocation.START)     # Kudo 2018
RobertaSpaceMarker  = BoundaryMarker("Ġ",       detached=True,  location=BoundaryMarkerLocation.START)     # Radford 2019
NoSpaceMarker       = BoundaryMarker("",        detached=False, location=BoundaryMarkerLocation.START)

IdentityPreprocessor = Preprocessor(IdentityMapper(), IdentityMapper(), IdentityPretokeniser())

TraditionalPretokeniser = PretokeniserSequence([
    WhitespacePretokeniser(destructive=True),
    PunctuationPretokeniser(HyphenMode.EXCLUDED)
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
    def __init__(self, marker: BoundaryMarker, byte_based: bool):
        super().__init__([
            WhitespacePretokeniser(destructive=True),
            AddWordBoundary(BoundaryMarker(" ", detached=True, location=marker.location)),
            PunctuationPretokeniser(HyphenMode.INCLUDED, group_adjacent_spaces_with_punctuation=marker.location),
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

class TruncateAndNormalise(MapperSequence):
    def __init__(self, truncate_after_chars: int):
        super().__init__([
            TruncateOnNearestWhitespace(truncate_after_chars),
            HuggingFaceNormaliser(tn.NFKC())
        ])


class ModernEnglishPretokeniser(PretokeniserSequence):
    """
    My common-sense pretokeniser can add any boundary marker for announcing words/punctuations.
    Good for modern-day subword systems like transformers.
    """
    def __init__(self, marker: BoundaryMarker, do_pseudobytes: bool=True, do_split_after_placing_boundaries: bool=True):
        whitespace_and_punctuation = [
            PunctuationPretokeniser(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
            WhitespacePretokeniser(destructive=True),
            EnglishApostrophes(do_nt=True)
        ]
        pseudos = [
            MapperAsPretokeniser(PseudoByteMapping())
        ]
        boundaries = [
            AddWordBoundary(marker)
        ]
        post_boundaries = [  # These split into pretokens that should NOT all have a word boundary. E.g.: for the string "2024 10 04", you do not want the pretokens ["2024", "10", "04"] to become ["_2", "_0", "_2", "_4", "_1", "_0", "_0", "_4"].
            IsolateDigits(),  # Also splits the word boundary off of the first digit.
            PunctuationPretokeniser(HyphenMode.ONLY)
        ]
        super().__init__(whitespace_and_punctuation + pseudos*do_pseudobytes + boundaries + post_boundaries*do_split_after_placing_boundaries)


class ModernEnglishPreprocessor(Preprocessor):
    def __init__(self, marker: BoundaryMarker, truncate_text_after_chars: int=1_000_000):
        super().__init__(
            TruncateAndNormalise(truncate_text_after_chars),
            IdentityMapper(),
            ModernEnglishPretokeniser(marker)
        )

CommonsensePreprocessor = ModernEnglishPreprocessor


class ModernEnglishPretokeniser_ByteCompatible(ModernEnglishPretokeniser):
    """
    Same as the ModernEnglishPretokeniser, except it is compatible with tokenisers that process their input
    in the byte domain (which usually happens by running bytes(..., "utf-8") on every pretoken).

    The issue with these tokenisers is two-fold:
        1. They map pseudo-byte characters, which each represent an atom of the alphabet, into even finer atoms that
           then need to be learnt as units, which takes up vocabulary space in tokenisers like BPE.
        2. In addition, exactly because these tokenisers cannot have any other atoms than the 256 byte units, they also
           don't support treating multi-byte boundary markers (e.g. Sennrich's multi-character </w> or Roberta's Ġ) as whole units.

    The solution to both of these is to set up the string such that .encode() will produce both of the effects we want:
        1. Don't apply a pseudo-byte mapping. Without adding characters, that means .encode() performs the same mapping we want.
        2. We pick a character that is mapped to a single atom, make sure it is removed from all pretokens, and then use
           that as a boundary marker at the front or back of pretokens. This means .encode() will map it to a single byte
           that appears nowhere else, and hence we effectively have a boundary marker. Since we start with whitespace
           tokenisation, whitespace is a suitable choice for that character.
    """
    def __init__(self, marker_location: BoundaryMarkerLocation):
        super().__init__(BoundaryMarker(substitute=" ", detached=True, location=marker_location), do_pseudobytes=False)


class ModernEnglishPreprocessor_ByteCompatible(ModernEnglishPreprocessor):
    """
    See explanation under ModernEnglishPretokeniser_ByteCompatible.
    """
    def __init__(self, marker: BoundaryMarker, truncate_text_after_chars: int=1_000_000):
        super().__init__(marker, truncate_text_after_chars)
        self.splitter = ModernEnglishPretokeniser_ByteCompatible(marker.location)

        self._marker = marker
        self._marker_space = BoundaryMarker(substitute=" ", detached=True, location=marker.location)
        self._pseudos = PseudoByteMapping()

    def pseudoToByteToken(self, pbyte_token: str) -> bytes:
        """
        Turns a token like "efficiÃ«ntie</w>", which represents a segment from a string to which we first applied
        a pseudo-byte mapping and then added a boundary marker, into a byte sequence where each character becomes exactly
        one byte (despite being Unicode characters of multiple bytes) and where the boundary marker is mapped to the byte
        corresponding to a space.

        This is good for cases where you have a tokeniser that needs a byte vocabulary, whilst you have a traditional
        pseudo-byte vocabulary at hand.
        """
        pseudobytes, marker_found = self._marker.isolate(pbyte_token)  # Is there a marker?
        if marker_found:
            pseudobytes = self._marker.concatenate(pseudobytes, self._pseudos.convert(" "))  # Put a space in place of the marker, as a pseudo-byte.

        return bytes(map(self._pseudos.PSEUDO_TO_BYTE.get, pseudobytes))

    def byteToPseudoToken(self, byte_token: bytes) -> str:
        """
        Turns a byte sequence that may include the byte representing a space
        into a pseudo-byte string with a proper space marker.

        This is good for cases where you have a vocabulariser or a tokeniser that produces bytes objects as tokens.
        """
        pseudobytes_with_space = "".join(map(self._pseudos.BYTE_TO_PSEUDO, byte_token))
        pseudobytes, marker_found = self._marker_space.isolate(pseudobytes_with_space)
        if marker_found:
            pseudobytes = self._marker_space.concatenate(pseudobytes, self._marker.substitute)

        return pseudobytes


class SentencePiecePreprocessor(ModernEnglishPreprocessor):
    """
    Preprocessor compatible with the SentencePiece package, which does its own preprocessing (both for BPE and for KudoPiece).
    In particular: you can give pretokens to SentencePiece by separating them by spaces, but SentencePiece will prefix each
    of them by its own boundary marker. Hence, you want a preprocessor that does not split into finer pretokens than possible given
    that all pretokens will receive a boundary marker.

    TODO: Wondering what to do with the marker substitute.
    """
    def __init__(self, marker: BoundaryMarker, truncate_text_after_chars: int=1_000_000):
        super().__init__(marker, truncate_text_after_chars)
        self.splitter = ModernEnglishPretokeniser(BoundaryMarker(substitute=" ", detached=True, location=marker.location),
                                                  do_split_after_placing_boundaries=False)
