from tktkt.wrappers.fallbackvocab import TokeniserWithByteFallback
from tktkt.factories.preprocessors import IdentityPreprocessor
from tktkt.models.greedy.directional import L2R_Greedy, Vocab
from tktkt.interfaces.identifiers import NoSpecials


def test_bytefallback():
    tk = L2R_Greedy(preprocessor=IdentityPreprocessor, vocab=Vocab(
        ["a", "b", "c", "L", "n"],
        specials=NoSpecials(),
        unk_id=0
    ))

    without_unk = "abc"
    with_unk = "abdefghijklm"
    print(tk.tokenise(without_unk), "aka", [tk.typeToId(t) for t in tk.tokenise(without_unk)])
    print(tk.tokenise(with_unk), "aka", [tk.typeToId(t) for t in tk.tokenise(with_unk)])

    print("Crude (fully UNK'ed):")
    tk_protected = TokeniserWithByteFallback(tk, crude_fallback=True)
    print(tk_protected.tokenise(without_unk))
    print(tk_protected.tokenise(with_unk))

    print("Aligned:")
    tk_protected = TokeniserWithByteFallback(tk, crude_fallback=False)
    print(tk_protected.tokenise(without_unk))
    print(tk_protected.tokenise(with_unk))

    # Some more complicated Unicode.
    with_unk = "L𝒶n𝒶"
    print(tk_protected.tokenise(with_unk))

    # Ambiguous UNKs
    with_unk = "azbzb"
    print(tk_protected.tokenise(with_unk))


if __name__ == "__main__":
    test_bytefallback()
