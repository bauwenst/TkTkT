"""
Some common instantiations of preparation functions.
"""
from ..interfaces.preprocessors import *
from ..preparation.splitters import *
from ..preparation.mappers import *
from ..preparation.huggingface import *

# Most common space markers
SennrichSpaceMarker = BoundaryMarker("</w>",    detached=False, location=BoundaryMarkerLocation.END)       # Sennrich 2016
IsolatedSpaceMarker = BoundaryMarker("[SPACE]", detached=True,  location=BoundaryMarkerLocation.ISOLATED)  # Huck 2017
KudoSpaceMarker     = BoundaryMarker("▁",       detached=True,  location=BoundaryMarkerLocation.START)     # Kudo 2018
RobertaSpaceMarker  = BoundaryMarker("Ġ",       detached=True,  location=BoundaryMarkerLocation.START)     # Radford 2019
NoSpaceMarker       = BoundaryMarker("",        detached=False, location=BoundaryMarkerLocation.START)
PrefixWhitespaceAsMarker = BoundaryMarker(" ",  detached=True,  location=BoundaryMarkerLocation.START)
SuffixWhitespaceAsMarker = BoundaryMarker(" ",  detached=True,  location=BoundaryMarkerLocation.END)

IdentityPreprocessor = Preprocessor(IdentityMapper(), IdentityMapper(), IdentityPretokeniser())


class BoundariesFromSpacesPretokeniser(PretokeniserSequence):
    """
    Generalisation of the RoBERTa/GPT-2 tokeniser. Principles it holds by:
        - Punctuation is always isolated from non-punctuation.
        - Word boundaries are only placed where there was a space originally.
    I recommend against using this. Use the ModernPretokeniser instead.

    The original implementation uses a start-of-word Ġ and is byte-based. For example, a sentence
        This is a (test) sentënce.
    becomes
        ĠThis // Ġis // Ġa // Ġ( // test // ) // ĠsentÃ«nce // .

    This implementation is more general than RoBERTa/GPT-2 since it allows customising the boundary marker and allows
    turning off the byte-based behaviour.
    """
    def __init__(self, marker: BoundaryMarker, byte_based: bool):
        super().__init__([
            OnWhitespace(destructive=True),
            AddWordBoundary(BoundaryMarker(" ", detached=True, location=marker.location)),
            IsolatePunctuation(HyphenMode.INCLUDED, group_adjacent_spaces_with_punctuation=marker.location),
            (MapperAsPretokeniser(PseudoByteMapping()) if byte_based else MapperAsPretokeniser(Replace(" ", "Ġ"))),
            MapperAsPretokeniser(Replace("Ġ", " ")),
            OnWhitespaceAndAddMarker(marker)
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


TraditionalPretokeniser = PretokeniserSequence([
    OnWhitespace(destructive=True),
    IsolatePunctuation(HyphenMode.EXCLUDED)
])

class TraditionalPreprocessor(Preprocessor):
    """
    Traditional, word-based preprocessor that just splits on spaces and non-hyphen punctuation.
    """
    def __init__(self, truncate_after_chars: int=1_000_000):
        super().__init__(
            uninvertible_mapping=TruncateAndNormalise(truncate_after_chars),
            invertible_mapping=IdentityMapper(),
            splitter=TraditionalPretokeniser()
        )


class ModernEnglishPretokeniser(PretokeniserSequence):
    """
    My common-sense pretokeniser can add any boundary marker for announcing words/punctuations.
    Good for modern-day subword systems like transformers.
    """
    def __init__(self, marker: BoundaryMarker, do_pseudobytes: bool=True, do_split_after_placing_boundaries: bool=True):
        whitespace_and_punctuation = [
            IsolatePunctuation(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
            OnWhitespace(destructive=True),
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
            IsolatePunctuation(HyphenMode.ONLY)
        ]
        super().__init__(whitespace_and_punctuation + pseudos*do_pseudobytes + boundaries + post_boundaries*do_split_after_placing_boundaries)


class ModernEnglishPreprocessor(Preprocessor):
    def __init__(self, marker: BoundaryMarker, truncate_text_after_chars: int=1_000_000):
        super().__init__(
            uninvertible_mapping=TruncateAndNormalise(truncate_text_after_chars),
            invertible_mapping=IdentityMapper(),
            splitter=ModernEnglishPretokeniser(marker)
        )

CommonsensePreprocessor = ModernEnglishPreprocessor  # Backwards-compatibility
Prefab1 = ModernEnglishPreprocessor

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


class SentencePiecePreprocessor_SpaceConcatenable(ModernEnglishPreprocessor):
    """
    Preprocessor compatible with the SentencePiece package, which does its own preprocessing (both for BPE and for KudoPiece).
    In particular: you can give pretokens to SentencePiece by separating them by spaces, but SentencePiece will prefix each
    of them by its own boundary marker. Hence, you want a preprocessor that does not split into finer pretokens than possible given
    that all pretokens will receive a boundary marker.

    Note: This is an imperfect approximation of the ModernEnglishPreprocessor. In particular, it doesn't split numbers
    and it doesn't split hyphenated words, which means hyphens will appear in subwords.
    Pretokens resulting from this pretokeniser can be concatenated with spaces and then fed into a SentencePiece vocabulariser.
    """
    def __init__(self, marker_location: BoundaryMarkerLocation, prefix_space_already_added: bool=False, truncate_text_after_chars: int=1_000_000):
        """
        :param prefix_space_already_added: Whether the tokeniser that follows after this is expected to add a space to
                                           every pretoken it receives or not. Normally, this should always be False; the
                                           only reason it would be True is if you accidentally set add_dummy_prefix to True
                                           when vocabularising, whose effect is to add a space in front of the sentences in
                                           the dataset during training (not in front of every pretoken, because the concept
                                           of "pretoken" doesn't exist in SentencePiece; there are uncrossable boundaries
                                           within a string, but it always remains one sentence) and this carries over to
                                           adding a space in front of every pretoken during inference because TkTkT sends
                                           individual pretokens to the tokeniser, which SP sees as one-word sentences.
        """
        marker = NoSpaceMarker if prefix_space_already_added else BoundaryMarker(substitute=" ", detached=True, location=marker_location)
        super().__init__(marker, truncate_text_after_chars)
        self.splitter = ModernEnglishPretokeniser(marker, do_split_after_placing_boundaries=False)


class ModernEnglishPretokeniser_SentencePieceCompatible(PretokeniserSequence):
    """
    Variant that is better suited for training with the SentencePiece package, which has two pretokenisation constraints
    to take into account (similar to the constraints taken into account for the byte-compatible preprocessor):
        - All spaces are converted to a "_" boundary marker, so the boundary marker added by the preprocessor must be a space.
        - The byte mapping must be such that all non-ASCII pseudo-byte characters, i.e. the characters that don't represent
          themselves but represent bytes of larger characters, can be grouped into one token, which is only possible when
          they belong to the same Unicode script. This is not the case for the HuggingFace pseudo-byte mapping, which includes
          characters like ½ and ÷.

    Pretokens resulting from this pretokeniser should be concatenated with an exotic Unicode character from a script
    known to SentencePiece that will never be used, and then fed into a SentencePiece vocabulariser. Do NOT concatenate
    these with spaces.

    This pretokeniser actually has a more generalisable ordering than the normal ModernEnglishPretokeniser:
        1. Split into coarse pretokens that should receive a boundary.
        2. To each pretoken, add a character (front, back or in between) that (1) will not interfere with the rest of the
           subpretokenisers that follow and (2) has an invertible result when sent through the alphabet mapping of choice.
           In this case, it's a space character.
        3. Split into fine pretokens to inhibit the tokeniser.
        4. Apply the alphabet mapping. You now have no guarantee that any of your usual splitters still works, and in fact,
           in this pretokeniser, hyphen and apostrophe splitting won't work anymore, for example. The ModernEnglishPretokeniser
           does not take this into account and just assumes that the alphabet mapping keeps hyphens and apostrophes intact.
        5. Invert the image of space characters under this mapping, and replace it by your word boundary of choice. In
           the case of SentencePiece, it must be a space because the built-in preprocessor turns spaces into boundaries itself.

    ---

    A note on vocabulariser-specific preprocessors:
    In an ideal world (which we aren't far from), every vocabulariser respects pretokens EXACTLY, i.e. no substrings
    are vocabularised that aren't present in the pretokens (unless single characters), and thus we can select our
    preprocessor P1 before selecting our vocabulariser.

    Sadly, with bad vocabularisers, our hand is forced. We're forced to rewrite our preprocessor (say, P2) such that
    it results in a hopefully isomorphic pretoken space. Then, you basically have three choices:
    1. Use the vocabulary as-is (in B space) and use P2+B as the effective preprocessor, like in training.
        - If the inferencer is good: use P2 + B, a small extra preprocessor that models the vocabulariser's built-in (the B is forcibly added because the vocabulary is in B space).
        - If the inferencer is bad: somehow change your preprocessor to be able to use the vocabulary.
            - If this is the original inferencer (e.g. SP ULM): use P2.
            - If this is another inferencer (e.g. SaGe with ULM vocab): use either
                - P2 + B + BB2 in the input and inv(B2) + inv(BB2) in the output, or
                - P2 + P2B2 in the input and inv(B2) + inv(P2B2) + B in the output.
    2. Use the vocabulary as-is (in B space), but use P1+P1B+B during inference as effective preprocessor, unlike the P2+B in training.
        - If the inferencer is good: use P1 + P1B + B. There is some expectation that P1 + P1B is almost exactly equivalent to P2, which means P1B could look like inv(P1) + P2.
        - If the inferencer is bad:
            - If this is the original inferencer (e.g. SP ULM): use P1 + P1B.
            - If this is another inferencer (e.g. SaGe with ULM vocab): use either
                - P1 + P1B + B + BB2 in the input and inv(B2) + inv(BB2) in the output, or
                - P1 + P1B2 in the input and inv(B2) + inv(P1B2) + P1B + B in the output.
    3. Adapt the vocabulary to P1 and use P1 as your preprocessor. I.e.: effective preprocessor in training was P2+B, but in inference it's truly P1.
        - If the inferencer is good: use P1.
        - If the inferencer is bad: use P1 + P1B in the input and inv(B) + inv(P1B) in the output
    If you assume that you only use good inferencers (e.g. use SentencePiece to make BPE and TkTkT to infer BPE),
    OR that the only bad inferencer you use is the one that accompanies the vocabulariser (e.g. using SPM BPE), in
    both cases these would be the preprocessors to write (other than P1 and P2):
        1. Only B
        2. P1B and B
        3. Nothing, but to initially convert the vocabulary, you effectively need
           inv(B) + inv(P1B), which are expected to be the same as inv(B) + inv(P2) + P1 (with the caveat that you
           may need to keep track of whether there was a boundary marker, since P1 will always add a boundary marker even if there wasn't any).
           So really, P1B and B as well.
    It is only in (3) that you can actually use the alphabet of your pre-written preprocessor P1.
    """
    def __init__(self, marker_location: BoundaryMarkerLocation, add_builtin_preprocessor: bool=False):
        """
        :param add_builtin_preprocessor: Whether to append SentencePiece's built-in preprocessor to the result.
                                         You want to do this if you want to use this preprocessor but not pass its results
                                         to a SentencePiece model, while still getting into the same preprocessing space.
        """
        self._location = marker_location
        space_marker = BoundaryMarker(substitute=" ", detached=True, location=marker_location)
        kudo_marker  = BoundaryMarker(substitute=KudoSpaceMarker.substitute, detached=True, location=marker_location)
        super().__init__([
            # Generate pretokens that need a boundary
            IsolatePunctuation(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
            OnWhitespace(destructive=True),  # Remove all spaces
            IsolateEnglishContractions(do_nt=True),

            # Add a space boundary, because we know what it becomes under the byte mapping.
            AddWordBoundary(space_marker),

            # Generate sub-pretokens
            IsolatePunctuation(HyphenMode.ONLY),
            IsolateDigits(),

            # Mapping
            MapperAsPretokeniser(LatinPseudoByteMapping()),  # Space markers become Ġ

            # Map image of spaces back to spaces, because SentencePiece will convert spaces to _
            MapperAsPretokeniser(ReplaceBoundary("Ġ", space_marker))
        ] + [
            MapperAsPretokeniser(ReplaceBoundary(" ", kudo_marker))
        ]*add_builtin_preprocessor)

    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        """
        The effective boundary marker of this SentencePiece pretokeniser is "_", because either add_builtin_preprocessor
        was true and it actually is, or you use it in front of a SentencePiece model and it converts to "_".
        """
        return BoundaryMarker(substitute=KudoSpaceMarker.substitute, detached=True, location=self._location)


class ModernEnglishPreprocessor_SentencePieceCompatible(Preprocessor):
    """
    For details, see the documentation for ModernEnglishPretokeniser_SentencePieceCompatible.
    """
    def __init__(self, marker_location: BoundaryMarkerLocation, add_builtin_preprocessor: bool=False, truncate_text_after_chars: int=1_000_000):
        super().__init__(
            uninvertible_mapping=TruncateAndNormalise(truncate_after_chars=truncate_text_after_chars),
            invertible_mapping=IdentityMapper(),
            splitter=ModernEnglishPretokeniser_SentencePieceCompatible(marker_location=marker_location, add_builtin_preprocessor=add_builtin_preprocessor)
        )

    def getBoundaryMarker(self) -> Optional[BoundaryMarker]:
        return self.splitter.getBoundaryMarker()


class Prefab2(Preprocessor):
    """
    Improvement upon Prefab1 in several regards:
        1. No more byte mapping, because it is quite useless in language modelling (who cares that you can represent
           a script you've never seen if you don't know what the representation in UTF-8 actually means).
        2. Capitals are no longer part of tokens, but handled by an extra [CAP] special.
        3. Support for non-English contractions.
        4. Limit on the amount of digits per token.
    """
    def __init__(self, marker: BoundaryMarker, truncate_text_after_chars: int=1_000_000):
        super().__init__(
            uninvertible_mapping=TruncateAndNormalise(truncate_text_after_chars),
            invertible_mapping=RegisterASCII(),
            splitter=PretokeniserSequence([
                IsolatePunctuation(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),  # TODO: This cannot discern "the number 123.567 is big" from "the number 123. 567 is big." so you need a rule that doesn't isolate punctuation when it is surrounded by digits.
                WhitespacePretokeniser(destructive=True),
                IsolateEnglishContractions(do_nt=True),
                PolariseApostrophes(tiebreak_left=True),
                # MapperAsPretokeniser(PseudoByteMapping())
                AddWordBoundary(marker),
                GroupDigits(n=3),  # TODO: This cannot handle Arabizi. So, a run of digits is not a number if it has a letter to its left or right, and should not be split up.
                IsolateConnectingHyphens(),
                AddCapitalMarker(ignore_marker=marker)
            ])
        )
