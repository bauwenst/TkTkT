from typing import List

from ...interfaces.tokeniser import TokeniserWithVocabDict, Preprocessor, Vocab
from .vocabularisation import SageVocabulariser


class SageTokeniser(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab):
        super().__init__(preprocessor, vocab)

        from sage_tokenizer.model import SaGeTokenizer
        self.backend = SaGeTokenizer(initial_vocabulary={
            SageVocabulariser._toHexString(t): i for t,i in vocab.items()
        })

    def tokenise(self, pretoken: str) -> List[str]:
        return [self.idToType(i) for i in self.backend.tokenize(pretoken, tokens_only=True)]
