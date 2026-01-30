"""
Tests related to whether type variables are filled properly.
"""
from typing import Generic, TypeVar


def theory_inheritance_fieldtype():
    T = TypeVar("T")

    class A(Generic[T]):
        def __init__(self, x: T):
            self._x = x


    class B(A[T]):
        def __init__(self, x: T, y: int):
            super().__init__(x=x)
            self._y = y


    class C(A, Generic[T]):  # Wrong way to do it
        def __init__(self, x: T, y: int):
            super().__init__(x=x)
            self._y = y


    class D(A):  # Wrong way to do it
        def __init__(self, x: T, y: int):
            super().__init__(x=x)
            self._y = y


    a = A("value")
    a._x
    b = B("value", 0)
    b._x
    c = C("value", 0)
    c._x
    d = D("value", 0)
    d._x


def theory_inheritance_methodtype():
    T = TypeVar("T")

    class Box(Generic[T]):
        def __init__(self, x: T):
            self._x = x

    class BoxMaker(Generic[T]):
        def __init__(self, y: T):
            self._y = y

        def makeBox(self) -> Box[T]:
            return self._makeBox()

        def _makeBox(self) -> Box[T]:
            return Box(self._y)

    class BoxMakerA(BoxMaker[T]):
        def _makeBox(self):
            return Box(self._y)

    class BoxMakerB(BoxMaker):  # Wrong way to do it (surprisingly!)
        def _makeBox(self):
            return Box(self._y)

    class BoxMakerC(BoxMaker, Generic[T]):  # Wrong way to do it (less surprising)
        def _makeBox(self):
            return Box(self._y)

    maker = BoxMakerA(y="thing")
    box = maker.makeBox()
    box._x

    maker = BoxMakerB(y="thing")
    box = maker.makeBox()
    box._x

    maker = BoxMakerC(y="thing")
    box = maker.makeBox()
    box._x


# def test_deserialiser():
#     from tktkt.factories.specials import RobertaSpecials
#     from tktkt.factories.artifacts import BPE32ki_SlimPajama3M
#
#     d1 = BPE32ki_SlimPajama3M(RobertaSpecials(-1,-1,-1,-1))
#     d1._specials  # Doesn't work, but only because of a bug in PyCharm that makes TypeVar behave badly when it has a bound.
#
#     d2: BPE32ki_SlimPajama3M[RobertaSpecials] = BPE32ki_SlimPajama3M(RobertaSpecials(-1,-1,-1,-1))
#     d2._specials  # Works
#
#     v = d2.getVocabulary()
#     v.specials  # Works

def test_extendedspecials():
    from tktkt.interfaces.identifiers import SpecialsExtended
    from tktkt.factories.specials import RobertaSpecials
    from tktkt.factories.artifacts import BPE32ki_SlimPajama3M

    extended_specials = SpecialsExtended(RobertaSpecials(-1,-1,-1,-1))
    extended_specials.specials  # FIXME: Doesn't work, oh oh

    d: BPE32ki_SlimPajama3M = BPE32ki_SlimPajama3M()
    v = d.getVocabulary(specials=extended_specials)
    v.specials

    # TODO: These work, but I'm quite disappointed by the fact that SpecialsExtended can't infer the type of WithSpecials due to it being in the constructor.
    extended_specials: SpecialsExtended[RobertaSpecials] = SpecialsExtended(RobertaSpecials(-1,-1,-1,-1))
    extended_specials.specials

    d: BPE32ki_SlimPajama3M = BPE32ki_SlimPajama3M()
    v = d.getVocabulary(specials=extended_specials)
    v.specials


def test_factory():
    from tktkt.factories.tokenisers import Factory_BPE
    f = Factory_BPE()
    t = f.buildTokeniser()
    t  # Works (tokeniser type is inferred)


def test_tokeniser():
    from tktkt.factories.tokenisers import Factory_BPE_Pythonic, BPE32ki_SlimPajama3M, ClassicBPE, TokeniserFactory
    from tktkt.factories.specials import RobertaSpecials
    from tktkt.interfaces.identifiers import SpecialsExtended

    s: SpecialsExtended[RobertaSpecials] = SpecialsExtended(RobertaSpecials(-1,-1,-1,-1), unk=0)

    d1: BPE32ki_SlimPajama3M = BPE32ki_SlimPajama3M()
    f1 = Factory_BPE_Pythonic(files=d1)
    t1 = f1.buildTokeniser()
    t1.vocab.specials  # Interestingly, the tokeniser type is inferred (because method types are no problem), but the bug in PyCharm prevents it from filling the TokeniserFactory's typevar using its constructor.

    d2 = BPE32ki_SlimPajama3M()
    f2: TokeniserFactory[ClassicBPE[RobertaSpecials]] = Factory_BPE_Pythonic(files=d2)
    t2 = f2.buildTokeniser()
    t2.vocab.specials  # Works

    d3 = BPE32ki_SlimPajama3M()
    f3 = Factory_BPE_Pythonic(files=d3)
    t3: ClassicBPE[RobertaSpecials] = f3.buildTokeniser()
    t3.vocab.specials  # Works
