from dataclasses import dataclass

from transformers import AutoTokenizer

from tktkt.factories.specials import RobertaSpecials
from tktkt.interfaces.identifiers import Specials, NoSpecials, AutoVocab, AutoVocabSpecs
from tktkt.interfaces import Vocab



@dataclass
class TextSpecials(Specials):
    CLS: int
    SEP: int
    PAD: int
    MASK: int


@dataclass
class ImageSpecials(Specials):
    IMG_START: int
    IMG_END: int


@dataclass
class NestedSpecials(Specials):
    text: TextSpecials
    image: ImageSpecials


@dataclass
class IllegalSpecials(Specials):
    basics1: TextSpecials
    basics2: TextSpecials



def tst_nested():
    s1 = TextSpecials(+1, +1, +1, +1)
    s2 = ImageSpecials(-1, -1)
    s3 = NestedSpecials(s1, s2)
    # IllegalSpecials(BasicSpecials(1,2,3,4), BasicSpecials(5,6,7,8))
    # print(list(s3.__iter_keys__()))

    v: Vocab[NestedSpecials] = Vocab(["a", "b", "c"], specials=s3)
    print(v.specials)
    print("CLS" in v)
    print("a" in v)
    print(v.size())
    print(list(v.specials))
    print(v)
    print(v.unsafe())


def tst_none():
    v: Vocab[NoSpecials] = Vocab(["a", "b", "c"], specials=NoSpecials())
    print(v.specials)
    print("CLS" in v)
    print("a" in v)
    print(v.size())
    print(list(v.specials))
    print(v)
    print(v.unsafe())


def tst_auto():
    v = AutoVocab.fromTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("roberta-base"),
        specials_specification=AutoVocabSpecs(
            specials_template=RobertaSpecials(BOS=-1, EOS=-1, PAD=-1, MASK=-1),
            special_to_string={
                "BOS": "<s>",
                "EOS": "</s>",
                "PAD": "<pad>",
                "MASK": "<mask>"
            }
        )
    )
    print(v)
    print(v.specials.PAD)


if __name__ == "__main__":
    tst_nested()
    print()
    tst_none()
    print()
    tst_auto()
