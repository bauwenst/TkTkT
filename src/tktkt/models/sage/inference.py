from pathlib import Path
from typing import List

from sage_tokenizer.model import SaGeTokenizer

from ...interfaces.tokeniser import TokeniserWithVocabDict, Preprocessor, Vocab
from .vocabularisation import SageVocabulariser


class SageTokeniser(TokeniserWithVocabDict):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab):
        super().__init__(preprocessor, vocab)
        self.backend = SaGeTokenizer(initial_vocabulary={
            SageVocabulariser._toHexString(t): i for t,i in vocab.items()
        })

    @classmethod
    def load(cls, preprocessor: Preprocessor, file_or_folder: Path):
        return cls(preprocessor, SageVocabulariser.load(file_or_folder, sorting_key=None, existing_types=None))

    def tokenise(self, pretoken: str) -> List[str]:
        return [self.idToType(i) for i in self.backend.tokenize(pretoken, tokens_only=True)]
