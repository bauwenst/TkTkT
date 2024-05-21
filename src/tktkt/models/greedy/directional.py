"""
Greedy tokenisers that choose their tokens by starting at one edge of the input and moving to the other edge.

According to HuggingFace, L2R_Greedy is the algorithm used by WordPiece for inference. This is not true for the
original WordPiece paper, but it is true for the BERT paper.

I add a right-to-left version and lazy variants of both (explained below). Also a left-to-right merge-based tokeniser.

Implementations adapted from my master's thesis (Bauwens, 2023). https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
"""
from typing import List

from ...interfaces.tokeniser import TokeniserWithVocabDict


class L2R_Greedy(TokeniserWithVocabDict):
    """
    Note that you can write a left-to-right tokeniser in two ways:
        - Lempel-Ziv-style, where you select progressively larger subwords and stop the moment you encounter a subword
          not in the vocab;
        - The inverse of that, i.e. you select progressively smaller subwords and stop the moment you encounter a subword
          that is in the vocab;

    This class implements the latter. It is "greedy" like regex operators by default: make the match as big as possible,
    even if that means you pass the first match to go look for longer matches while meanwhile passing over strings that don't match.
    """

    def tokenise(self, word: str) -> List[str]:
        tokens = []
        token_length = len(word)
        while word:
            if token_length == 0 or word[:token_length] in self.vocab:
                if token_length == 0:  # This means the previous iteration, 'word[:1] in self.vocab' was False. Hence, there are no tokens for the remaining word. Pretend like the character does belong to the vocab. (Will be UNK'ed as a type.) TODO: Alternative is to produce self.unk.
                    token_length = 1

                tokens.append(word[:token_length])
                word = word[token_length:]
                token_length = len(word)
            else:
                token_length -= 1

        return tokens


class L2R_Lazy(TokeniserWithVocabDict):
    """
    "Lazy" like a lazy regex operator: matches as few characters as possible in order to have a match, i.e. find a string
    unknown to the vocabulary, at which point the token so far will be grouped.

    In the edge case that a single-character token is already unknown to the vocabulary, it is segmented off, but will
    be mapped to an UNK ID when looked up in the vocabulary.
    (The alternative, i.e. keep looking for a 2-character, 3-character, ... token that is known and then stopping when
    you again find a string that is unknown, probably won't work well, since that single character probably also won't
    exist inside bigger types.)
    """

    def tokenise(self, word: str) -> List[str]:
        tokens = []

        token_start  = 0
        token_length = 0
        while token_start < len(word):
            # Make a token if extending the length of the current token by 1 (looking ahead one position) gives an unknown token, or reaches past the end of the string (i.e. so far the token is in the vocab, and there are no more characters to add).
            if token_start+token_length == len(word) or word[token_start:token_start+token_length+1] not in self.vocab:
                if token_length == 0:  # This means you don't even recognise the current character. Pretend like you recognised the isolated character as a token. TODO: Alternatively, append self.unk.
                    token_length = 1

                tokens.append(word[token_start:token_start+token_length])
                token_start += token_length
                token_length = 0
            else:
                token_length += 1

        return tokens


class R2L_Greedy(TokeniserWithVocabDict):

    def tokenise(self, word: str) -> List[str]:
        tokens = []
        token_length = len(word)
        while word:
            if token_length == 0 or word[len(word)-token_length:] in self.vocab:
                if token_length == 0:
                    token_length = 1
                tokens.append(word[len(word)-token_length:])
                word = word[:len(word)-token_length]
                token_length = 0
            else:
                token_length -= 1

        tokens.reverse()
        return tokens


class R2L_Lazy(TokeniserWithVocabDict):

    def tokenise(self, word: str) -> List[str]:
        tokens = []

        token_end    = len(word)  # exclusive
        token_length = 0
        while token_end > 0:
            if token_end-token_length == 0 or word[token_end-token_length-1:token_end] not in self.vocab:
                if token_length == 0:
                    token_length = 1

                tokens.append(word[token_end-token_length:token_end])
                token_end -= token_length
                token_length = 0
            else:
                token_length += 1

        tokens.reverse()
        return tokens


class Xu(TokeniserWithVocabDict):
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
