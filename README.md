# TkTkT
A collection of Pythonic subword tokenisers.

## Pronunciation
The acronym stands for ToKeniser ToolKiT and is supposed to be pronounced fast and with beatbox hi-hats
(kind of like "tuh-kuh-tuh-kuh-ts" but as fast as you can). It is mandatory that you do this, because I said so.

## Installation
1. Create an editable install for [`bpe_knockout`](https://github.com/bauwenst/BPE-knockout) using the instructions given in its repo.
2. Install [`fiject`](https://github.com/bauwenst/fiject#installation).
3. Follow the exact same instructions as for `fiject`, but for this repo.

## Architecture
The goal of TkTkT is to provide a straightforward Pythonic interface for everything-tokenisation, and to be as object-oriented
as possible. The main interfaces are found under `tktkt.interfaces`. 

Fundamentally, all tokenisers are a `Tokeniser` that have a `Preprocessor`.

- The `Tokeniser` class has two important methods: 
  - `.tokenise(pretoken: str) -> List[str]`: segments a string as-is into parts.
  - `.prepareAndTokenise(text: str) -> List[str]`: applies the tokeniser's preprocessor and then tokenises each pre-token separately.

- The `Preprocessor` class is a pipeline of three components: a non-invertible text mapping, an invertible text mapping, 
  and a pretokeniser that splits strings into smaller strings.

## Examples
### KudoPiece (ULM)
Let's say you want to train and load an English ULM tokeniser, which is notorious for being a convoluted process. 
In TkTkT, that would go like this (note that ULM is called "KudoPiece" in TkTkT because it is a less ambiguous name):
```python
from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser
from tktkt.preparation.instances import IdentityMapper, AppendSpace, IdentityPretokeniser, Preprocessor

def load(model_path: Path):    
    preprocessor = Preprocessor(
        IdentityMapper(), 
        AppendSpace(front_not_back=True), 
        IdentityPretokeniser()
    )
    return KudoPieceTokeniser(preprocessor, model_path)


from tktkt.models.kudopiece.training import *
from string import ascii_letters

def train(sentence_corpus: Iterable[str]):
    args_alpha = KudoPieceArguments_Alphabet(
        required_chars=[l for l in ascii_letters], 
        byte_fallback=True, 
        character_coverage=0.9995
    )
    args_algo = KudoPieceArguments_Algorithm()

    trainer = KudoPieceTrainer(
        word_boundary_location=SpaceMarkerLocation.START,
        final_vocab_size=40_000,
        alphabet_arguments=args_alpha,
        algorithm_arguments=args_algo,
        file_stem="kudopiece_en"
    )
    return trainer.train_from_iterator(sentence_corpus, strings_need_space_splitting=True)


if __name__ == "__main__":
    your_corpus = ...

    model_path = train(your_corpus)
    ## The location of your model will look like this:
    # from tktkt.files.paths import DataPaths
    # model_path = DataPaths.pathToModels() / "kudopiece_en" / "kudopiece_en_xxxx-yy-zz_aa-bb-cc.model"
    tk = load(model_path)

    print(tk.prepareAndTokenise("Hello there, my good friend!"))
```

## Why does this exist if we have HuggingFace `tokenizers`?
First of all, note that *TkTkT* has backwards compatibility with HuggingFace `tokenizers`. There are wrapper classes for
tokenisers and pretokenisers under `tktkt.models.huggingface`.

Here's a non-exhaustive list of reasons:
- The HuggingFace `tokenizers` library has horrifically un(der)documented Python interfaces, so programming with it is
  a nightmare. 
- The `tokenizers.pre_tokenizers` submodule has so much technical debt that it can't be patched. Some examples:
    - The mapping from Unicode codepoints to UTF-8 bytes, as first used in GPT-2, is only implemented in the `ByteLevel` 
      pretokeniser. Yet, it is concerned with more than this, since it splits on spaces and punctuation (optionally prefixed 
      by a space) before applying the mapping. This is wrong for at least three reasons: users of the byte mapping don't
      necessary want the string to be split, it synonymises prefixed spaces (converted to `Ġ`) with start-of-word boundaries 
      whilst actually all words (even those directly preceded by punctuation) should be marked with such a boundary, and
      it assumes that such boundaries should always be at the start of a word.
    - The GPT-2 convention of having a word boundary at the *start* of (almost) all words is hardcoded throughout
      `transformers` and `tokenizers` (with options that commonly look like `add_prefix_space`) even though the original
      BPE paper used word boundaries at the *end* of words (`</w>`). Only supporting the start-of-word convention is bad 
      because this deteriorates downstream performance for e.g. Germanic languages, where a compound has its head at the
      end and hence it should be allowed to tokenise the head with the exact same tokens as it would be if it was isolated. 
- Weird holdovers like the `Precompiled` normaliser that allow even less insight into what's happening.
- In the little documentation that does exist (e.g. for WordPiece and KudoPiece), there are so many 
  theoretical inaccuracies that we shouldn't even have confidence in anything that isn't a BPE tokeniser implemented by them. 
  Their [explanation for KudoPiece](https://huggingface.co/learn/nlp-course/chapter6/7), an algorithm which itself was 
  already poorly explained originally, is so wrong it is actually painful.
- They offer very few core models (basically only BPE and KudoPiece, which [`sentencepiece`](github.com/google/sentencepiece) already offers
  and keeps much more updated)
  whilst there exist many more in the literature, and the likelihood that someone who knows the literature comes along to
  implement all of them in C++ is rather low.
