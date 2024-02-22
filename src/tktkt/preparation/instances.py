"""
Some common instantiations of preparation functions.
"""
from .spacemarking import SpaceMarker, SpaceMarkerLocation
from .splitters import WhitespaceAndMarkerPretokeniser
from ..interfaces.preparation import Preprocessor
from .mappers import IdentityMapper, tn, Normaliser, PseudoByteMapping

SennrichSpaceMarker = SpaceMarker("</w>", detached=False, location=SpaceMarkerLocation.END)
RobertaSpaceMarker = SpaceMarker("Ġ", detached=True, location=SpaceMarkerLocation.START)
IsolatedSpaceMarker = SpaceMarker("[SPACE]", detached=True, location=SpaceMarkerLocation.TOKEN)

RobertaPretokeniser = WhitespaceAndMarkerPretokeniser(RobertaSpaceMarker)
RobertaPreprocessor = Preprocessor(Normaliser(tn.NFKC()), PseudoByteMapping(), RobertaPretokeniser)
