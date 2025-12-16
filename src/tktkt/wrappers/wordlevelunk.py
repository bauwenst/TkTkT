from ..interfaces.tokenisers import *


class TokeniserWithWordLevelUnk(TokeniserWithVocabulary[WithSpecials]):
    """
    When an UNK appears anywhere in the tokeniser's output for a given word, this wrapper replaces that entire word
    by one big UNK, to avoid having partial tokens for that word. (You lose information this way, but maybe you like that.)
    """

    def __init__(self, tokeniser_with_unks: TokeniserWithVocabulary[WithSpecials]):
        self.core = tokeniser_with_unks
        super().__init__(preprocessor=self.core.preprocessor, vocab=self.core.vocab)

    def tokenise(self, pretoken: str) -> Tokens:
        tokens = self.core.tokenise(pretoken)
        if any(self.typeToId(token) == self.vocab.UNK for token in tokens):
            return [""]  # TODO: Actually, you can't implement this method at the string level. It happens purely at the ID level.
        else:
            return tokens
