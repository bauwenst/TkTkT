# TkTkT
A collection of Pythonic subword tokenisers.

## Pronunciation
The acronym stands for ToKeniser ToolKiT and is supposed to be pronounced fast and with beatbox hi-hats
(kind of like "tuh-kuh-tuh-kuh-ts" but as fast as you can). It is mandatory that you do this, because I said so.

## Why does this exist if we have HuggingFace `tokenizers`?
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
- In the little documentation that does exist (e.g. for WordPiece and ULM), there are so many 
  theoretical inaccuracies that we shouldn't even have confidence in anything that isn't a BPE tokeniser implemented by them. 
  Their explanation for ULM, which was already poorly explained originally, is so wrong it is actually painful.
- They offer very few core models (basically only BPE and KudoPiece, which [`sentencepiece`](github.com/google/sentencepiece) already offers)
  whilst there exist many more in the literature, and the likelihood that someone who knows the literature comes along to
  implement all of them in C++ is rather low.
