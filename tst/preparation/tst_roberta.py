from tst.preamble import *

from transformers import RobertaTokenizer
robbert_tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")

from tktkt.interfaces.preprocessors import Preprocessor
from tktkt.preparation.splitters import *
from tktkt.preparation.mappers import *
from tktkt.preparation.huggingface import HuggingFacePretokeniser
from tktkt.factories.preprocessors import RobertaPreprocessor, RobertaSpaceMarker, SennrichSpaceMarker, \
    IsolatedSpaceMarker, ModernEnglishPreprocessor
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser


def robbertsucks():
    s = " a flatscreen is not a (flatscreen) but it is a flat screen."
    print(s)
    print(robbert_tokenizer.tokenize(s))

    print(robbert_tokenizer.tokenize("this has     many     spaces"))

    from tktkt.preparation.mappers import PseudoByteMapping
    mapping = PseudoByteMapping()
    print(list(map(mapping.BYTE_TO_PSEUDO.get, mapping.SPACING_BYTES)))


def punctuation():
    splitter = IsolatePunctuation(HyphenMode.ONLY)
    example = "energie-efficiëntie, dat is cool!"
    print(splitter.split(example))

    splitter = IsolatePunctuation(HyphenMode.EXCLUDED)
    print(splitter.split(example))

    splitter = IsolatePunctuation(HyphenMode.INCLUDED)
    print(splitter.split(example))


def steps():
    word     = "Pℛ𝒪𝒥ë𝒞𝒯"
    sentence = "It's a Pℛ𝒪𝒥ë𝒞𝒯, bruh! (low-key triggered)"
    clean = RobertaPreprocessor.irreversible.convert(sentence)

    s = ModernEnglishPreprocessor(marker=RobertaSpaceMarker)
    print(s.do(word))
    print(s.do(sentence))
    print()

    print(RobertaPreprocessor.do(word))
    print(RobertaPreprocessor.do(sentence))
    print()

    bytemapped = RobertaPreprocessor.reversible.convert(clean)
    print(clean)
    print(bytemapped)
    print(RobertaPreprocessor.splitter.split(bytemapped))
    # print(tokeniseAsWord(word, tokeniser=))


def roberttest():
    mine = RobertaPreprocessor
    hf   = Preprocessor(splitter=HuggingFacePretokeniser.fromFullTokeniser(robbert_tokenizer))

    s = "hello (this is an) ex-ample...! Hawai'i, I've missed "
    print(mine.do(s))
    print(hf.do(s))


def backendcall():
    s = " Hello supercalifragilistic (this is an) ëx-ample...! Hawai'i, I've missed 的 a lot "

    old_pre = HuggingFacePretokeniser.fromFullTokeniser(robbert_tokenizer)
    print("Pretokens original:", old_pre.split(s))
    print("->", robbert_tokenizer.tokenize(s))

    from tokenizers.pre_tokenizers import Sequence
    dummy_pre = Sequence([])
    robbert_tokenizer.backend_tokenizer.pre_tokenizer = dummy_pre
    print("Pretokens dummy:", dummy_pre.pre_tokenize_str(s))
    print("->", robbert_tokenizer.tokenize("Ġ" + s))  # Even though the byte-based pretokeniser has been removed, it is still aware of the byte alphabet (possibly due to what's in the vocab) and deletes every character it doesn't recognise. No UNKs, just delete.

    print("With dummy, applying .tokenize() on original pretokens:")
    all_tokens = []
    for pretoken in old_pre.split(s):
        all_tokens.extend(robbert_tokenizer.tokenize(pretoken))
    print("->", all_tokens)


def compareWrapper():
    s = " Hello supercalifragilistic (this is an) ëx-ample...! Hawai'i, I've missed 的 a lot "

    mine = HuggingFaceTokeniser(robbert_tokenizer)
    print(robbert_tokenizer.tokenize(s))
    print(mine.prepareAndTokenise(s))
    print(mine.tokenise(s.replace(" ", "Ġ")))
    print(mine.tokenise(s))


if __name__ == "__main__":
    robbertsucks()
    # roberttest()
    # backendcall()
    # compareWrapper()
