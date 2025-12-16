"""
Tokeniser which imagines BPE merges on the fly by checking if a pair can be merged into a type part of the vocabulary.
"""
from math import inf, isinf

from ...interfaces.tokenisers import *
from ...wrappers.multiplexing import SuccessionalTokeniser


class MergelessBPE(TokeniserWithVocabulary[WithSpecials], SuccessionalTokeniser):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials]):
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self._boundary = preprocessor.getBoundaryMarker()

    def _initialTokens(self, pretoken: str) -> Tokens:
        return self._boundary.atomise(pretoken)

    def _finalTokens(self, tokens: Tokens) -> Tokens:
        tokens = list(tokens)
        candidates = ["".join(pair) for pair in zip(tokens[:-1], tokens[1:])]  # a b c d e f has candidates [ab, bc, cd, de, ef]. If you pick cd, the candidates become [ab, bcd, cde, ef]
        while candidates:
            index_of_highest_priority = min(range(len(candidates)), key=lambda i: self.vocab.get(candidates[i], inf))
            if isinf(index_of_highest_priority):
                return tokens

            new_token = candidates.pop(index_of_highest_priority)  # candidates = [ab, bc, de, ef]
            tokens[index_of_highest_priority] = new_token          # tokens = [a, b, cd, d, e, f]
            tokens.pop(index_of_highest_priority+1)                # tokens = [a, b, cd, e, f]

            if index_of_highest_priority > 0:
                candidates[index_of_highest_priority-1] = tokens[index_of_highest_priority-1] + new_token  # candidates = [ab, bcd, de, ef]
            if index_of_highest_priority < len(tokens) - 1:
                candidates[index_of_highest_priority] = new_token + tokens[index_of_highest_priority+1]  # candidates = [ab, bcd, cde, ef]

        return tokens
