from ...interfaces.tokenisers import *
from .vocabularisation import SageVocabulariser


class SageTokeniser(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials]):
        super().__init__(preprocessor=preprocessor, vocab=vocab)

        from sage.tokeniser import SageTokenizer
        self.backend = SageTokenizer(initial_vocabulary={
            SageVocabulariser._toHexString(t): i for t,i in vocab.items()
        }, add_alphabet=True)

    def tokenise(self, pretoken: str) -> Tokens:
        return [self.idToType(i) for i in self.backend.pretokenize_and_tokenize(pretoken)]
