from tktkt.models.ngram.base import *
from tktkt.evaluation.morphological import tokeniseAndDecode


if __name__ == "__main__":
    from itertools import product
    sentence = "Life could be a drëam!"
    word = "supercälifragïlisticëxpialidocious"

    Ns = [2,3,4,5,10]
    modes = [NgramByteBasedMode.CHAR_NGRAMS, NgramByteBasedMode.CHAR_NGRAMS_AS_BYTES, NgramByteBasedMode.BYTE_NGRAMS]
    for n,m in product(Ns, modes):
        tk = NgramTokeniser(n, m)

        print(n,m)
        print(tk.prepareAndTokenise(sentence))
        print(tokeniseAndDecode(sentence, tk))
        print(tk.preprocessor.undo(tk.prepareAndTokenise(sentence)))
        print()
