<img src="./doc/logo.png">

# TkTkT: the ToKeniser ToolKiT
A collection of Pythonic subword tokenisers and text preprocessing tools, with full
backwards- *and* forwards-compatibility with HuggingFace `tokenizers`. One package to rule them all.

Quick navigation:
- <a href="#installation">Installation</a>
- <a href="#features">Features</a>
- <a href="#examples">Examples</a>
- <a href="#architecture">Architecture</a>

## Features
### Supported tokenisers
All subword tokenisers are defined under `tktkt.models`. Many of these can be instantiated without much background knowledge using the factory classes in `tktkt.factories`.
Also, **any HuggingFace tokeniser** can be wrapped into a TkTkT tokeniser, and **any TkTkT tokeniser** can be wrapped into a HuggingFace tokeniser.

Currently, the package implements:
- Byte-pair encoding (BPE) tokenisers:
  - Classical **BPE** ([Sennrich et al., 2016](https://aclanthology.org/P16-1162/)), with added support for any word boundary marker (`Ġ`, `_`, `</w>`, ...) and *n*-ary merges (byte-tuple encoding, BTE).
  - **BPE-dropout** ([Provilkov et al., 2020](https://aclanthology.org/2020.acl-main.170/))
  - **BPE-knockout** ([Bauwens & Delobelle, 2024](https://aclanthology.org/2024.naacl-long.324/))
  - **PickyBPE** ([Chizhov et al., 2024](https://aclanthology.org/2024.emnlp-main.925/))
  - **ScaffoldBPE** ([Lian et al., 2025](https://dl.acm.org/doi/10.1609/aaai.v39i23.34633))
  - **TrimmedBPE** ([Cognetta et al., 2024](https://arxiv.org/abs/2404.00397))
  - Other experimental variants I implemented just for fun:
    - **BPE-breakdown**: BPE which starts randomly undoing merges after it finishes deterministically, similar to [StochasTok](https://arxiv.org/abs/2506.01687).
    - **Non-geometric BPE-dropout**: BPE-dropout, but rather than picking merges geometrically, picks them uniformly.
    - **EnsuredBPE**: BPE where the last merges have been replaced by the merges necessary to ensure that a given list of strings is in the vocabulary.
    - **ShuffledBPE**: BPE but with merge priorities shuffled, although types are never shuffled to a priority before the ancestors in their merge tree.
- **Unigram language model (ULM)**, dubbed *KudoPiece* in TkTkT ([Kudo, 2018](https://aclanthology.org/P18-1007/)):
  - Wrapper around the [SentencePiece](https://github.com/google/sentencepiece) package, or
  - Native implementation in TkTkT
- Greedy tokenisers:
  - **MaxMatch** ([Hiraoka, 2022](https://aclanthology.org/2022.coling-1.430)), a.k.a. **left-to-right greedy** tokenisation, and also **right-to-left** ([Bauwens, 2023](https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf) and later [Uzan et al., 2024](https://arxiv.org/abs/2403.01289))
  - **FLOTA** ([Hofmann et al., 2022](https://aclanthology.org/2022.acl-short.43/)), i.e. random-access longest-first tokenisation. 
  - Other experimental variants:
    - **Last-BPE-first**: random-access youngest-first tokenisation (specifically for BPE vocabularies).
    - **Left-to-right-to-left greedy**: L2R2L_Greedy
- **GRaMPa** ([Bauwens et al., 2025](https://aclanthology.org/2025.acl-long.1180/)): randomised segmentation constrained by a vocabulary.
- **SaGe** ([Yehezkel & Pinter, 2023](https://aclanthology.org/2023.eacl-main.45/)) vocabularisation.
- **Derivative leverager (DeL)** ([Hofmann et al., 2021](https://aclanthology.org/2021.acl-long.279/)), both training and segmentation.
- Other, less interesting tokenisers:
  - Character/byte **N-grams**.
  - **Lempel-Ziv-Welch (LZW)** as a tokeniser ([Zouhar et al., 2023](https://aclanthology.org/2023.acl-long.284/)).

Currently work in progress:
- Morfessor family
- VOLT

#### Multiplexing
TkTkT is the only package that supports **multiplexing** multiple tokenisers into one big tokeniser that alternates 
between each of them. There are multiplexers that do this deterministically (e.g. choosing the tokeniser that compresses
the input the most) or stochastically (e.g. choosing among a set of tokenisers uniformly).

### Evaluation metrics
TkTkT's evaluation framework aims to do as little work as possible. It can dispatch tokens produced by a tokeniser
to as many metrics as you need at once, and caches everything it can so you don't have to compute any metric twice.
See [here](https://github.com/LAGoM-NLP/ConfoundingFactors/blob/2170fc0137fee2fda2705afeaf1931651c73f74c/scripts/perplexity/av.py#L116) for an example.

TkTkT currently supports the following intrinsic tokeniser evaluation metrics:
- **Fertility** statistics: how many tokens the tokeniser produces per word, and how many segmentations its vocabulary could produce in theory.
- **Morphological** boundary recognition: using the tokeniser as a binary classifier for whether two morphemes meet at each
  position in a word.
- **Information-theoretic** measures, including *Rényi entropy* and *Rényi efficiency*.
- **Window-based** metrics like MATTR.
- **Bigram metrics** to quantify the richness of token contexts, like *accessor variety*.
- **Comparisons** between two tokenisers: how much they tokenise words exactly the same, and how much their split points overlap.

### Security
It should be impossible for users to jailbreak a language model by forcing its tokeniser to produce a special token (e.g. `<|endoftext|>`, system prompt delimiters, ...).
The reason other packages do not have this guarantee is that they represent special tokens as strings and then give the
tokeniser access to these strings. In TkTkT, **subword vocabularies are objects** that hide special tokens from their
tokeniser. In fact, **special tokens are defined as integers**, not as strings.

When loading a tokeniser trained in another package that probably inserted specials into the vocabulary, TkTkT explicitly 
requires the user to declare which strings are actually specials, and hides them from the tokeniser.

### User-friendliness
#### Caching
When a tokeniser finishes training, you shouldn't be forced to mess with file paths to connect your training and
testing scripts together. In TkTkT, training code caches its results. When you rerun it, it will load its
results from disk and skip the waiting time.

#### Type-checking
I really f\*cking passionately hate when my IDE cannot perform autocompletion because of poor type annotation or design. 
People have been suffering under the idiocy of `AutoTokenizer` for too long. In TkTkT, everything is as type-annotated as possible, which means:
- There are no checkpoint strings in TkTkT. There are `Artifacts` objects, which not only declare how to get the results
  of training, but also which `Preprocessor` was used, so that complex preprocessing objects are just known and never have to be stringified.
- When loading a tokeniser using a `TokeniserFactory` and `Artifacts` (the equivalent of `AutoTokenizer` of a `str`),
  the exact type of tokeniser is known.
- The special tokens in a vocabulary appear in autocompletion. For example, for BERT's BPE tokeniser, `tokeniser.vocab.specials.`
  will show `CLS`, `SEP`, `PAD`, `MASK` in your IDE.
- After training a tokeniser, the results on disk are already parsed into an object for you. When you train a BPE tokeniser,
  the `results` that come out aren't a dumb file path. It's an `Artifacts` object that has a `.getVocabulary()` and `.getMerges()` method.

### Preprocessing
TkTkT has a rich set of text mappings and pretokenisers that preprocess text before it is tokenised, including
support for stochastic perturbation. Unlike other libraries, **preprocessors are objects, not regular expressions.**
This allows much more powerful processing than regex, whilst being more easy to read. See if you can understand 
this arguably complicated transformation:

```python
from tktkt.preparation.splitters import *
from tktkt.preparation.mappers import PseudoByteMapping
from tktkt.factories.preprocessors import RobertaSpaceMarker


class ExamplePretokeniser(PretokeniserSequence):
    def __init__(self):
        super().__init__([
            IsolatePunctuation(HyphenMode.EXCLUDED, protect_apostrophes_without_spaces=True),
            OnWhitespace(destructive=True),
            IsolateEnglishContractions(do_nt=True),

            MapperAsPretokeniser(PseudoByteMapping()),
            AddWordBoundary(RobertaSpaceMarker),

            IsolateDigits(),
            IsolatePunctuation(HyphenMode.ONLY)
        ])
```

TkTkT also comes with language-specific pretokenisation like Japanese word segmentation and Thai word segmentation.

### Visualisers
The following tokenisation procedures can be visualised:
- BPE/BTE: the final merge tree (in regular LaTeX), as well as an animated progression of the merges (in LaTeX Beamer).

### Opinionated
Apart from the type-checking and caching described above, TkTkT enforces several truths about tokenisation which are not present in other packages:
- Tokenisers produce string segments (_tokens_), not integer identifiers (_IDs_). There is no "BPE way" of mapping `["_un", "believ", "able"]`
  to integers. That should be done by a separate object, the `Vocab`.
- Preprocessors should not (only) be regular expressions. They should be chains of Python code.
- Spaces should not be treated as word boundaries. If a word does not have a "prefix space", e.g. because it is the start of the sentence
  or because it is preceded by a punctuation mark, it should _still_ receive a boundary character, and the developer should be
  able to decide if that boundary comes at the start or the end of the word.
- Special tokens (`CLS`, `SEP`, `BOS`, ...) should not have a string representation (`"[CLS]", "[SEP]", "<s>", ...`), or at least they should not be in the vocabulary, 
  or at least they should not be used by the tokeniser. All of these are violated by HuggingFace `transformers` (see the bottom of this README).
- Preprocessing should not be baked into the tokeniser.
  - Tokenisers should segment the given string, not transform it into some unpredictably other string thing and segment that one.
  - Any parameters regarding preprocessing should not be in the tokeniser's constructor, except for the actual `Preprocessor` itself.
- Preprocessors should not be pickled and base64'ed into opaque objects. They should be defined as Python code.

## Architecture
### Main interfaces
The goal of TkTkT is to provide a straightforward Pythonic interface for everything-tokenisation, and to be as 
object-oriented as possible. The main interfaces are found under `tktkt.interfaces`. 

#### Inference
Fundamentally, all tokenisers are a `Tokeniser` that have a `Preprocessor`.

- The `Tokeniser` class has two important methods: 
  - `.tokenise(pretoken: str) -> List[str]`: segments a string as-is into parts.
  - `.prepareAndTokenise(text: str) -> List[str]`: applies the tokeniser's preprocessor and then tokenises each pre-token separately.

- The `Preprocessor` class is a pipeline of three components: 
  1. a non-invertible text mapping
  2. an invertible text mapping
  3. a pretokeniser that splits strings into smaller strings.

To map tokens (string segments) to identifiers (integers) for indexing into an embedding matrix, this interface is
extended to the `TokeniserWithVocabulary`. This class makes use of a `Vocab` object which keeps _types used by the tokeniser_ 
separated from the _specials used by a downstream language model_. This prevents token injection attacks, which 
HuggingFace `transformers`/`tokenizers` is vulnerable to (see the bottom of this README).

#### Training
To learn the parameters of a `Tokeniser` (e.g. BPE merges), there is the `Vocabulariser` class.
It can learn from word-count files or from corpora of sentences. It takes a `Preprocessor` exactly like `Tokeniser`,
except `Vocabulariser` is for training the tokeniser (vocabularisation) and `Tokeniser` is for inference (segmentation).

#### Loading
To make it easier to load the results of a vocabularisation run from storage back into Python, there are `Artifacts` classes
to do this for you. 

For ease-of-use, many `Tokeniser` classes have a `TokeniserFactory` defined for them that simplify the instantiation process.
Often, a `TokeniserFactory` will take an `Artifacts` object to provide it any files.

### Submodules
The packages is divided into the following submodules:
- `tktkt.interfaces`: contains the main parent classes from which all other classes derive. 
  - The most important classes are `TextMapper`, `Pretokeniser`, `Preprocessor`, `Vocabulariser`, `Tokeniser`, `Artifacts`, and `TokeniserFactory`.
- `tktkt.preparation`: contains all the text preprocessing tools.
- `tktkt.models`: contains all the tokenisation (i.e. vocabularisation and/or segmentation) algorithms.
- `tktkt.evaluation`: contains procedures with which to quantify a `Tokeniser` through inference.
- `tktkt.factories`: contains a bunch of pre-defined constructor calls, for both vocabularies and tokenisers:
  - `tktkt.factories.artifacts`: contains classes that load the files for specific tokenisers.
  - `tktkt.factories.tokenisers`: contains tokeniser factories.
  - `tktkt.factories.specials`: contains sets of special identifiers (CLS, SEP, BOS, EOS, ...).
  - `tktkt.factories.preprocessors`: contains a bunch of pre-defined preprocessors so you don't have to.
    Check out the `ModernEnglishPreprocessor`, for example.
  - `tktkt.factories.evaluation`: contains pre-built tokeniser evaluation pipelines.
- `tktkt.wrappers`: contains classes that wrap around existing tokenisers to equip them with more features.
  - `tktkt.wrappers.multiplexing`: alternate between multiple tokenisers within the same sentence.
  - `tktkt.wrappers.hashingvocab`: add a string-to-integer mapping to a `Tokeniser` that can produce any substring, turning it into a `TokeniserWithFiniteIdRange`.
- `tktkt.visualisation`: contains procedures to generate explanatory LaTeX code about some models.
- `tktkt.util`: contains tools peripheral to tokenisation, like string formatting, combinatoric calculations, iterable functions, timing, etc...

## Installation
Simply run
```shell
pip install "tktkt[github] @ git+https://github.com/bauwenst/TkTkT"
```
where you should leave out the `[github]` suffix only if you have editable installations of any of my other packages, 
like [`bpe_knockout`](https://github.com/bauwenst/BPE-knockout) (but you probably don't).

## Examples
### HuggingFace compatibility
In the example below, a BPE tokeniser is loaded from the HuggingFace hub as a `PreTrainedTokenizerFast` and converted into a TkTkT `Tokeniser` object.
Then, this object is itself converted into a HuggingFace `PreTrainedTokenizer` again.
```python
# Backwards-compatibility:
from transformers import AutoTokenizer
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser

hf_roberta = AutoTokenizer.from_pretrained("roberta-base")
tktkt_roberta = HuggingFaceTokeniser(hf_roberta)

###
sentence = " That's so supercalifragilisticexpialidocious, Günther!"
print("Full tokenisation pipeline:")
print("\tHF Tk:", hf_roberta.tokenize(sentence))  # Note the lack of autocompletion on this.
print("\tTkTkT:", tktkt_roberta.prepareAndTokenise(sentence))
print("Only the preprocessing:")
print("\tTkTkT:", tktkt_roberta.preprocessor.do(sentence))
###

# Forwards-compatibility:
from tktkt.interfaces.huggingface import TktktToHuggingFace

hf_tktkt_roberta = TktktToHuggingFace(tktkt_roberta, specials_from=hf_roberta)
print(hf_tktkt_roberta.tokenize(sentence))
```

### Training and instantiating BPE
Here's a minimal working example to train a BPE tokeniser on the first 100 000 examples of an English Wikipedia dataset:

```python
from datasets import load_dataset
from tktkt.factories.preprocessors import ModernEnglishPreprocessor, KudoSpaceMarker
from tktkt.models.bpe.vocabularisation import BPEVocabulariser

corpus        = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True).take(100_000)
preprocessor  = ModernEnglishPreprocessor(marker=KudoSpaceMarker)
vocabulariser = BPEVocabulariser(preprocessor=preprocessor, vocab_size=32_768)
results       = vocabulariser.vocabulariseFromHf(corpus, text_field="text")
```
That's _**just 7 lines of code to get a tokeniser from a corpus!**_ To load the result into a HuggingFace-accelerated tokeniser, we can call

```python
from tktkt.models.huggingface.bpe import HuggingFaceBPETokeniser

tokeniser = HuggingFaceBPETokeniser(
    preprocessor=results.preprocessorEffective(),
    vocab=results.getVocabulary(),
    merges=results.getMerges()
)
```
The best part? You can put this in the same script as the `vocabulariser.vocabulariseFromHf`, right below it.
All training methods are cached, which means that if the result was computed before, it will be loaded from disk immediately.

If you want to load the result of training in a different project, the way to do it is to first write a `BPEArtifacts` class
that knows which preprocessor you used and where to find your results. Once you have this, use the `Factory_BPE` to
read the artifacts for you.

As an example: I once trained BPE across the first 3 million examples in SlimPajama. The `BPEArtifacts` class for this
specific training run are predefined in TkTkT:

```python
from tktkt.factories.artifacts import BPE32ki_SlimPajama3M
from tktkt.factories.tokenisers import Factory_BPE

tokeniser = Factory_BPE(files=BPE32ki_SlimPajama3M()).buildTokeniser()
```
Note again that the preprocessor comes with the artifacts, so the factory doesn't require that you specify it.

### Training and instantiating ULM (a.k.a. KudoPiece)
Let's now say you want to train and load an English ULM tokeniser. You are, of course, scared of the `sentencepiece` library
because its Python interface is a thin wrapper around a command-line call, not allowing autocompletion in your IDE.
In TkTkT, you would proceed as follows (note that ULM is called "KudoPiece" in TkTkT because many tokenisers are based on a language model of unigrams).

First we instantiate a preprocessor, and call the trainer with relevant training arguments. Then, we load the saved
results into a tokeniser.

```python
from tktkt.factories.preprocessors import ModernEnglishPreprocessor_SentencePieceCompatible, BoundaryMarkerLocation
from tktkt.models.kudopiece.vocabularisation import *

### Your data iterator goes here.
sentence_corpus: Iterable[str] = ...
###

preprocessor = ModernEnglishPreprocessor_SentencePieceCompatible(
    marker_location=BoundaryMarkerLocation.START
)

trainer = KudoPieceVocabulariser(
    preprocessor=preprocessor,
    final_vocab_size=40_000,
    arguments=KudoPieceArguments(character_coverage=0.9995),
    file_stem="tutorial"
)
results = trainer.vocabulariseFromHf(sentence_corpus, text_field="text")


from tktkt.models.kudopiece.segmentation import KudoPieceTokeniser

tokeniser = KudoPieceTokeniser(preprocessor=results.preprocessorNative(), model_file=results.getModelFile())
print(tokeniser.prepareAndTokenise("Hello there, my good friend!"))
```

### Custom preprocessing
TkTkT preprocesses text into pretokens _not_ with a regular expression, but with a sequence of Python objects that can
perform any operation they want on the current pretokens. It is hence strictly more expressive than regex-based pretokenisation.
For example:

```python
from tktkt.factories.preprocessors import *

toy_preprocessor = Preprocessor(
    Lowercaser(),
    Replace("!", "."),
    PretokeniserSequence([
        OnWhitespace(),
        IsolatePunctuation(),
        AddWordBoundary(KudoSpaceMarker)
    ])
)

print(toy_preprocessor.do("This example will be preprocessed (even without a tokeniser)!"))
```
This can then be used to instantiate any TkTkT tokeniser, whose functionality is decoupled from the preprocessor. For example:
```python
from tktkt.models.greedy.directional import L2R_Greedy, Vocab
from tktkt.factories.specials import NoSpecials

tokeniser = L2R_Greedy(
    preprocessor=toy_preprocessor,
    vocab=Vocab(
        ["a", "b", "c", "d", "ab", "ba", ".", ",", "▁"],
        specials=NoSpecials(),
        unk_id=0
    )
)

print(tokeniser.prepareAndTokenise("A bad cab, ABBA!"))
print(tokeniser.tokenise("abc."))
```
There are many more preprocessing classes available, some pre-made. Check out the `ModernEnglishPreprocessor` 
for typical modern use-cases.

## Why does this package exist if we have HuggingFace `tokenizers`?
First of all, note again that TkTkT has backwards compatibility with HuggingFace `tokenizers`. 
There are wrapper classes for tokenisers under `tktkt.models.huggingface` and for normalisers/pretokenisers under
`tktkt.preparation.huggingface`.

Note also that the `tokenizers` package actually lacks most of the logic which the HuggingFace ecosystem uses for tokenisation.
Both the main tokenisation interface (`PretrainedTokenizer`/`PretrainedFastTokenizer`) as well as deserialisation (`AutoTokenizer`) 
live in `transformers`, not `tokenizers`. So let's consider the union of both.

Here's a non-exhaustive list of reasons:
1. The `tokenizers` package offers very few core models (basically only BPE and KudoPiece, which [`sentencepiece`](github.com/google/sentencepiece) already offers
   and actually keeps updated), yet there exist many more in the literature. The likelihood that someone who knows the 
   literature comes along to implement all of them in Rust or C++ is rather low.
2. The HuggingFace `tokenizers` package has horrifically un(der)documented Python interfaces.
   - Some `tokenizers` classes even accept arguments that aren't in their signature. 
   - Your IDE cannot know what type the result of `AutoTokenizer` is, and thus you can never have autocompletion on it.
3. The `tokenizers` package is implemented in Rust. This has at least two big issues with it:
   - There is no possibility of inspecting implementations in any Python IDE. 
   - Some of the responsibilities of existing Python code in `transformers` for tokenisation are unfortunately delegated to the Rust
     codebase when it is used, and thus re-implemented in `tokenizers`. This means HuggingFace should maintain parallel
     implementations, but of course, Python and Rust engineers are different people, and thus the codebases have
     become desynchronised. So, some constructor arguments do work for some tokenisers whilst doing nothing for others.
   - One example of a bug: the `tokenizers` package's RoBERTa BPE implementation [removes the highest-priority merge from the tokeniser](https://github.com/huggingface/transformers/blob/9b5a6450d481b0f02834684ffd8b3ba4cbbd6fe0/src/transformers/models/roberta/tokenization_roberta.py#L194)
     unless the merge file is preceded by a `#version` tag? This doesn't conform to [the BPE standard](https://github.com/rsennrich/subword-nmt/), and almost cost me a paper.
4. The `transformers` package forces special tokens (`<|endoftext|>`, `[CLS]`, `[SEP]`, ...) to be treated as if they
   are user input. That's a security vulnerability.
   - Special tokens should _never_ be treated like text. They should be seen as IDs without a name. They are purely for
     adding extra embedding vectors to the input of a downstream language model.
   - Yet, in the `PreTrainedTokenizerBase` class, specials must be declared using _only a string_ with _no identifier_,
     and the point at which these strings receive their identifier is when they are run through the _exact same_ method
     that converts the tokens from user input to identifiers.
5. The `tokenizers` interface does not allow separating preprocessing from the actual tokenisation algorithm.
   - The `PreTrainedTokenizerBase` class, from which the "slow" (Pythonic) `PreTrainedTokenizer` and "fast" (Rustic) 
      `PreTrainedTokenizerFast` classes both inherit, only declares an end-to-end `.tokenize()` method (equivalent to TkTkT's
      `.prepareAndTokenise()`). The interface for these subclasses is different enough that both lack features of the other: 
      - Whereas `PreTrainedTokenizer` does declare a `._tokenize()` (equivalent to TkTkT's `.tokenise()`), I challenge you
        to find the equivalent for `PreTrainedTokenizerFast`. Best you'll find is `.backend_tokenizer.model.tokenize()`, 
        which outputs unusable objects of class `tokenizers.Token`.
      - Whereas `PreTrainedTokenizerFast` has fields `.backend_tokenizer.pre_tokenizer` and `.backend_tokenizer.normalizer`
        (untyped of course, so you can't get autocompletion on their methods unless you manually assign them to a variable
        and annotate it yourself), `PreTrainedTokenizer` has [no access to a pretokeniser](https://github.com/huggingface/transformers/issues/26254).
        Preprocessing has to be defined inside `._tokenize()`, which means you're doing two steps of preprocessing (one inside `.tokenize()`
        and one inside `._tokenize()`) making this `._tokenize()` no longer equivalent to TkTkT's `.tokenise()`.
      - For `PreTrainedTokenizerFast`, the `.backend_tokenizer.pre_tokenizer` and `.backend_tokenizer.normalizer` fields 
        can both be `None`, rather than an identity transform like in TkTkT, meaning you always have to check if they exist. 
        Also funny: even when they are not `None`, you can't check if they exist with a simple `if t.backend_tokenizer.normalizer: ...`
        because somehow that's always `False`.
   - Also, the `PreTrainedTokenizerBase` interface is not defined with `@abstractmethod` but with an ever-increasing 
      amount of `raise NotImplementedError` methods. In other words: it's hard to know which methods need to be implemented
      and there's no enforcement mechanism to ensure everything has been implemented.
6. Some preprocessors are literally pickled (i.e. the Python object in memory is converted to bytes), encoded in base64, and
   saved to `tokenizer.json`. The result is a completely opaque black-box preprocessor (which `tokenizers` just calls "`Precompiled`").
   - You have no access to its internals in Python, it has no interface, and without the original training script and
     the exact versions of all the packages involved, you will never find a description of what it does. This is probably 
     a holdover from adapting the `sentencepiece` package to HuggingFace, but it doesn't have to be that way. TkTkT has transparent preprocessors.
7. The `tokenizers.pre_tokenizers` submodule has weird technical debt.
      - The one way to apply GPT-2's mapping from Unicode codepoints to UTF-8 bytes is to use the `ByteLevel` 
        pretokeniser. What does it do? Much more than just the GPT-2 mapping: it first splits the text in front of each 
        spaces and punctuation mark (optionally prefixed by a space). This is wrong for at least three reasons: 
        - Users of the byte mapping don't necessary want the string to be split;
        - It synonymises prefixed spaces (converted to `Ġ`) with start-of-word boundaries 
          whilst actually all words (even those directly preceded by punctuation) should be marked with such a boundary; 
        - It assumes that such boundaries should always be at the start of a word.
      - The GPT-2 convention of having a word boundary at the *start* of (almost) all words is hardcoded throughout
        `transformers` and `tokenizers` (with options that commonly look like `add_prefix_space`) even though the original
        BPE paper used word boundaries at the *end* of words (`</w>`). Only supporting the start-of-word convention is bad 
        because this deteriorates downstream performance for e.g. Germanic languages, where a compound has its head at the
        end and hence it should be allowed to tokenise the head with the exact same tokens as it would be if it was isolated.
8. In the little HuggingFace tokeniser documentation that does exist, there are so many 
    theoretical inaccuracies (e.g. for WordPiece and KudoPiece/ULM) that we shouldn't even have confidence in anything 
    that isn't a BPE tokeniser implemented by them. 
    Their [explanation for KudoPiece](https://huggingface.co/learn/nlp-course/chapter6/7), an algorithm which itself was 
    already poorly explained originally, is mathematically absurd.

There is also the [pyonmttok](https://github.com/OpenNMT/Tokenizer) package which has better design than `tokenizers`, but also sticks to BPE and KudoPiece.

## Pronunciation
The acronym stands for ToKeniser ToolKiT and is supposed to be pronounced fast (kind of like "tuh-kuh-tuh-kuh-ts" but as fast as you can). It is mandatory that you do this.

If you are Brazilian, you may pronounce it "tuca tuca" while playing [the official TkTkT theme song](https://open.spotify.com/track/2aX7w5bdbES8A9H5FDydSA)
(yes, the demented state of modern-day tokeniser implementations will leave you with an equally demented taste in music; but [I dance salsa](https://bauwenst.github.io/categories/tutorials/dancing/) on the weekends).
