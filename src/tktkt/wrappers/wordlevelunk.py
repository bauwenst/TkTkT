from typing import List

from ..interfaces.tokeniser import TokeniserWithVocabDict


class TokeniserWithWordLevelUnk(TokeniserWithVocabDict):
    """
    When an UNK appears anywhere in the tokeniser's output for a given word, this wrapper replaces that entire word
    by one big UNK, to avoid having partial tokens for that word. (You lose information this way, but maybe you like that.)
    """

    def __init__(self, tokeniser_with_unks: TokeniserWithVocabDict):
        self.core = tokeniser_with_unks
        super().__init__(self.core.preprocessor, self.core.vocab, self.core.unk)

        self.unk_id = self.typeToId(self.unk)

    def tokenise(self, pretoken: str) -> List[str]:
        tokens = self.core.tokenise(pretoken)
        if any(self.typeToId(token) == self.unk_id for token in tokens):
            return [self.unk]
        else:
            return tokens
