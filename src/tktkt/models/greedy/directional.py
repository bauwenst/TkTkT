"""
Greedy tokenisers that choose their tokens by starting at one edge of the input and moving to the other edge.

According to HuggingFace, L2R_Greedy is the algorithm used by WordPiece for inference. This is not true for the
original WordPiece paper, but it is true for the BERT paper.

I add a right-to-left version and lazy variants of both (explained below). Also a left-to-right merge-based tokeniser.

Implementations adapted from my master's thesis (Bauwens, 2023). https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
"""
from typing import List

from ...interfaces.tokeniser import TokeniserWithVocab


class L2R_Greedy(TokeniserWithVocab):
    """
    Note that you can write a left-to-right tokeniser in two ways:
        - Lempel-Ziv-style, where you select progressively larger subwords and stop the moment you encounter a subword
          not in the vocab;
        - The inverse of that, i.e. you select progressively smaller subwords and stop the moment you encounter a subword
          that is in the vocab;

    I've implemented the latter. It finds bigger subwords at the start, obviously.
    """

    def tokenise(self, word: str) -> List[str]:
        tokens = []
        i = len(word)
        while word:
            if word[:i] in self.vocab:
                tokens.append(word[:i])
                word = word[i:]
                i = len(word)
            else:
                i -= 1

        return tokens


class L2R_Lazy(TokeniserWithVocab):

    def tokenise(self, word: str) -> List[str]:
        tokens = []
        i = 1  # Assumes letters are in the vocab. Otherwise, the loop will run forever by adding empty strings the whole time.
        while word:
            if word[:i] not in self.vocab or i > len(
                    word):  # The "or" is justified because you've reached past the max length, the word hasn't been consumed, and the "not in vocab" hasn't been triggered, so the full string is in the vocab.
                tokens.append(word[:i - 1])
                word = word[i - 1:]
                i = 1
            else:
                i += 1

        return tokens


class R2L_Greedy(TokeniserWithVocab):

    def tokenise(self, word: str) -> List[str]:
        tokens = []
        i = 0
        while word:
            if word[i:] in self.vocab:
                tokens.append(word[i:])
                word = word[:i]
                i = 0
            else:
                i += 1

        tokens.reverse()
        return tokens


class R2L_Lazy(TokeniserWithVocab):

    def tokenise(self, word: str) -> List[str]:
        tokens = []
        i = len(word) - 1
        while word:
            if word[i:] not in self.vocab or i == -1:
                tokens.append(word[i + 1:])
                word = word[:i + 1]
                i = len(word) - 1
            else:
                i -= 1

        tokens.reverse()
        return tokens


class Xu(TokeniserWithVocab):
    """
    Models the inference method described in the VOLT paper (Xu 2021). It might just be an incorrect description of
    BPE, but it's worth actually attempting this weird algorithm:

    > After generating the vocabulary, VOLT uses a greedy strategy to encode text similar to BPE. To encode text,
    > it first splits sentences into characterlevel tokens. Then, we merge two consecutive tokens into one token if
    > the merged one is in the vocabulary. This process keeps running until no tokens can be merged. Out-of-vocabulary
    > tokens will be split into smaller tokens.

    So, it's a left-to-right approach, but only merges pairs. I also do a second iteration over the string.
    For example:

        a b c d e f g
        ab c d e f g
        abc d e f g
        abc de fg
        abcde fg

    Quite similar to left-to-right smallest-first.
    """

    def tokenise(self, word: str) -> List[str]:
        tokens = list(word)
        changed = True
        while changed:
            i = 0
            changed = False
            while i < len(tokens) - 1:
                merged = tokens[i] + tokens[i + 1]
                if merged in self.vocab:
                    tokens[i:i + 2] = [merged]
                    changed = True
                else:
                    i += 1
        return tokens
