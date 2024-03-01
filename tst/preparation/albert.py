"""
Possibly the only mainstream model that uses the Kudo 2018 tokeniser (which I'll call KudoPiece) is ALBERT.

Since you can't access its configuration from Python, you have to inspect the config:
    https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json

The normaliser is a sequence of:
    Replace("``", '"')
    Replace("''", '"')
    NFKD
    StripAccents
    Lowercase
    Precompiled
where the latter is apparently some weird undocumented thing that exists as a C++ adapter for the SentencePiece library.
    https://huggingface.co/docs/transformers.js/api/tokenizers#tokenizersprecompiled--code-normalizer-code

The pretokeniser is a sequence of:
    WhitespaceSplit
    Metaspace("▁", add_prefix_space=true)
"""
from tktkt.models.huggingface.wrapper import HuggingFaceTokeniser

from transformers import AutoTokenizer
from transformers.models.albert.tokenization_albert_fast import AlbertTokenizerFast


def tst_equivalence_wrappedPreprocessor():
    s = " The Fibonacci       reconnaissance unit did not acquiesce adequately to the pernicious circumstances."
    tk: AlbertTokenizerFast = AutoTokenizer.from_pretrained("albert/albert-base-v2")

    wrapped_tk = HuggingFaceTokeniser(tk, for_single_words=False)

    print("Sentence:     ", s)
    print("Original norm:", tk.backend_tokenizer.normalizer.normalize_str(s))
    print("TkTkT norm:   ", wrapped_tk.preprocessor.irreversible.convert(s))
    print()

    print("Original prep:", [t for t,_ in tk.backend_tokenizer.pre_tokenizer.pre_tokenize_str(tk.backend_tokenizer.normalizer.normalize_str(s))])
    print("TkTkT prep:   ", wrapped_tk.preprocessor.do(s))
    print()

    print("Original full:", tk.tokenize(s))
    print("TkTkT full:   ", wrapped_tk.prepareAndTokenise(s))
    print()

    print("Original undo:", tk.convert_tokens_to_string(tk.tokenize(s)))
    print("TkTkT undo:   ", wrapped_tk.preprocessor.undo(wrapped_tk.prepareAndTokenise(s)))
    print()

    print("Original undo_per_token:", [tk.backend_tokenizer.decoder.decode([t]) for t in tk.tokenize(s)])  # The ULM decoder is probably something like "".join(tokens).replace("_", " ").strip(), which means that spaces are only protected when they appear in tokens that are surrounded by other tokens.
    print("TkTkT undo_per_token:   ", wrapped_tk.preprocessor.undo_per_token(wrapped_tk.prepareAndTokenise(s)))
    print()

    print("Original undo pairs:", [tk.backend_tokenizer.decoder.decode(pair) for pair in zip(tk.tokenize(s)[::2], tk.tokenize(s)[1::2])])
    print("TkTkT undo pairs:   ", [wrapped_tk.preprocessor.undo(pair) for pair in zip(tk.tokenize(s)[::2], tk.tokenize(s)[1::2])])


if __name__ == "__main__":
    tst_equivalence_wrappedPreprocessor()