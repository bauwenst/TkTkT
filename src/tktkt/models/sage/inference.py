from ...interfaces.tokenisers import *
from .vocabularisation import SageVocabulariser


class SageTokeniser(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials]):
        super().__init__(preprocessor=preprocessor, vocab=vocab)

        from sage_tokenizer.model import SaGeTokenizer
        self.backend = SaGeTokenizer(initial_vocabulary={
            SageVocabulariser._toHexString(t): i for t,i in vocab.items()
        })

    def tokenise(self, pretoken: str) -> Tokens:
        return [self.idToType(i) for i in self.backend.tokenize(pretoken, tokens_only=True)]
